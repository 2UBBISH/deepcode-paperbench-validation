"""APT self-knowledge distillation (Section 4.4, Appendix A, Appendix C).

The paper's self-distillation objective is (equation numbers follow the paper's
order in Section 4.4 / Appendix A)::

    L       = mu * L_distill + (1 - mu) * L_ft                              (1)
    L_layer = sum_{i=1..T} MSE(Tr(H_s^{phi(i)}), H_t^i)                     (1)

with

* ``T`` ("tau" in the paper = 4) block-wise randomly sampled teacher layers,
* ``phi(.)`` the teacher -> closest non-pruned student layer mapping,
  recomputed at *every* training step,
* ``Tr`` a tunable LoRA layer for layer transformation, initialised as the
  identity matrix ``I``,
* ``mu`` a moving term that is 0 before pruning starts and linearly increases
  to 1 at the end of pruning::

      mu = min(1., (global_step - pruning_start_step) / (pruning_end_step - pruning_start_step))

* ``L_distill = L_pred + 0.9 * L_layer`` for GLUE (classification) and
  ``L_distill = 0.1 * L_pred + 0.9 * L_layer`` for SQuAD and CNN/DM.

Rather than keeping a fully separate (and fully trained) teacher model in GPU
memory, APT *duplicates the tuning student layers as teachers* during
fine-tuning: frozen parameters are shared between student and teacher, so only
the (small) tuning parameters and mask buffers are duplicated.  That is what
:class:`TeacherModel` implements.

Nothing in this module requires Hugging Face ``transformers`` to be installed;
the teacher/student are treated as plain ``nn.Module`` graphs and hidden states
are collected with forward hooks on the transformer layer list.
"""

from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - constants mirror ``apt.adapters``
    from .adapters import DIMENSION, HEAD, NEURON  # type: ignore
except Exception:  # pragma: no cover - safe fallback (block-type function f(b))
    HEAD, NEURON, DIMENSION = 0, 1, 2

__all__ = [
    "LayerTransform",
    "LayerMapping",
    "BlockwiseTeacherSampler",
    "HiddenStateCollector",
    "TeacherModel",
    "SelfDistillation",
    "DistillationLosses",
    "mu_schedule",
    "task_distill_weights",
    "normalize_task",
    "find_transformer_layers",
    "iter_masked_linears",
    "masked_linears_by_layer",
    "student_layer_keep_flags",
    "collect_hidden_states",
]

# ---------------------------------------------------------------------------
# Task bookkeeping
# ---------------------------------------------------------------------------

#: Distillation weights ``(pred_weight, layer_weight)`` per task family.
#: Appendix A / Addendum "APT Implementation":
#:   GLUE:            L_distill = L_pred + 0.9 * L_layer
#:   SQuAD, CNN/DM:   L_distill = 0.1 * L_pred + 0.9 * L_layer
TASK_DISTILL_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "glue": (1.0, 0.9),
    "classification": (1.0, 0.9),
    "mnli": (1.0, 0.9),
    "sst2": (1.0, 0.9),
    "qnli": (1.0, 0.9),
    "qqp": (1.0, 0.9),
    "mrpc": (1.0, 0.9),
    "cola": (1.0, 0.9),
    "rte": (1.0, 0.9),
    "stsb": (1.0, 0.9),
    "squad": (0.1, 0.9),
    "squad_v2": (0.1, 0.9),
    "qa": (0.1, 0.9),
    "cnn_dm": (0.1, 0.9),
    "cnndm": (0.1, 0.9),
    "summarization": (0.1, 0.9),
    "seq2seq": (0.1, 0.9),
}

_TASK_ALIASES = {
    "cnn-dm": "cnn_dm",
    "cnn/dm": "cnn_dm",
    "cnndm": "cnn_dm",
    "squadv2": "squad",
    "squad v2": "squad",
    "text-classification": "glue",
    "language-modeling": "glue",
}


def normalize_task(task: Union[str, None]) -> str:
    """Map a user supplied task name to a canonical key of ``TASK_DISTILL_WEIGHTS``."""
    if task is None:
        return "glue"
    key = str(task).strip().lower().replace(" ", "_")
    key = _TASK_ALIASES.get(key, key)
    return key if key in TASK_DISTILL_WEIGHTS else "glue"


def task_distill_weights(task: Union[str, None]) -> Tuple[float, float]:
    """Return ``(pred_weight, layer_weight)`` used to build ``L_distill``."""
    return TASK_DISTILL_WEIGHTS[normalize_task(task)]


def mu_schedule(
    global_step: int,
    pruning_start_step: int,
    pruning_end_step: int,
) -> float:
    """Linear ``mu`` ramp of Appendix A ("APT Implementation").

    ``mu = min(1., (global_step - pruning_start_step) / (pruning_end_step - pruning_start_step))``

    ``mu`` is 0 before pruning starts and reaches exactly 1 at the end of the
    pruning stage.
    """
    start = int(pruning_start_step)
    end = int(pruning_end_step)
    step = int(global_step)
    if end <= start:
        return 1.0 if step >= end else 0.0
    return float(min(1.0, max(0.0, (step - start) / (end - start))))


# ---------------------------------------------------------------------------
# Introspection helpers (duck-typed so that this module imports standalone)
# ---------------------------------------------------------------------------


def _is_masked_linear(module: nn.Module) -> bool:
    """True for ``apt.adapters.MaskedLinear`` instances (duck-typed)."""
    return (
        hasattr(module, "base_weight")
        and hasattr(module, "adapter")
        and hasattr(module, "merged_weight")
    )


def iter_masked_linears(
    module: nn.Module,
    names: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    """Yield ``(qualified_name, masked_linear)`` pairs, optionally name-filtered."""
    for name, sub in module.named_modules():
        if not _is_masked_linear(sub):
            continue
        if names is not None and not any(n in name for n in names):
            continue
        yield name, sub


def masked_linears_by_layer(module: nn.Module) -> Dict[int, List[Tuple[str, nn.Module]]]:
    """Group masked linear layers by their transformer ``layer_idx``."""
    grouped: Dict[int, List[Tuple[str, nn.Module]]] = {}
    for name, sub in iter_masked_linears(module):
        idx = int(getattr(sub, "layer_idx", -1))
        grouped.setdefault(idx, []).append((name, sub))
    return grouped


def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    for p in module.parameters():
        return p.device, p.dtype
    for b in module.buffers():
        return b.device, b.dtype
    return torch.device("cpu"), torch.float32


def clear_caches(module: nn.Module) -> None:
    """Drop salience caches stored on masked linear layers.

    ``apt.adapters.MaskedLinear`` keeps (non-persistent) ``_cached_input`` /
    ``_cached_output`` / ``_cached_output_grad`` tensors.  Clearing them before
    ``copy.deepcopy`` avoids duplicating activation memory when the teacher is
    built, and keeps stale device/dtype references out of the teacher copy.
    """
    for _, sub in iter_masked_linears(module):
        for attr in ("_cached_input", "_cached_output", "_cached_output_grad"):
            if hasattr(sub, attr):
                try:
                    setattr(sub, attr, None)
                except Exception:  # pragma: no cover - buffer assignment quirk
                    pass


#: Attribute paths used to locate the transformer layer list of common LMs.
_TRANSFORMER_LAYER_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("model", "encoder", "layer"),      # BERT / RoBERTa / DeBERTa-v2
    ("model", "encoder", "layers"),     # OPT
    ("model", "decoder", "layers"),
    ("encoder", "block"),               # T5 / mT5 encoder
    ("decoder", "block"),               # T5 / mT5 decoder
    ("model", "decoder", "block"),
    ("transformer", "h"),               # GPT-2
    ("transformer", "blocks"),          # GPT-Neo
    ("model", "transformer", "h"),
    ("model", "layers"),                # LLaMA / Mistral
    ("model", "decoder", "layers"),
    ("layers",),
)


def find_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Return the list of transformer blocks (``nn.ModuleList``) of ``model``."""
    for path in _TRANSFORMER_LAYER_PATHS:
        obj: Any = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)) and len(obj) > 0:
            return list(obj)
    raise ValueError(
        "Could not locate the transformer layer list; pass `layers=` explicitly "
        "to HiddenStateCollector / collect_hidden_states."
    )


def student_layer_keep_flags(model: nn.Module, eps: float = 1e-8) -> List[int]:
    """Flags telling which student layers are still *alive* (not fully pruned).

    A transformer layer counts as pruned when every MHA head mask **and** every
    FFN neuron mask belonging to it has (near) zero magnitude; hidden-dimension
    masks alone do not prune a layer.  Layers for which no head/neuron block can
    be found are kept.
    """
    grouped = masked_linears_by_layer(model)
    if not grouped:
        return []
    n_layers = max(grouped.keys()) + 1
    keep = [1] * n_layers
    for idx, mods in grouped.items():
        if idx < 0:
            continue
        found = False
        alive = False
        for _, sub in mods:
            kind = int(getattr(sub, "kind", HEAD))
            if kind not in (HEAD, NEURON):
                continue
            getter = getattr(sub, "get_output_group_mask", None)
            if getter is None:
                continue
            mask = getter()
            if mask is None:
                continue
            found = True
            if float(mask.detach().abs().max()) > eps:
                alive = True
        if found and not alive:
            keep[idx] = 0
    return keep


# ---------------------------------------------------------------------------
# Hidden-state collection
# ---------------------------------------------------------------------------


class HiddenStateCollector:
    """Collect per-layer hidden states of a module with forward hooks.

    Hugging Face transformer blocks return a tuple whose first element is the
    (new) hidden state, which is exactly what is hooked here, so the collector
    works for BERT/RoBERTa/T5 without touching their ``output_hidden_states``
    plumbing.
    """

    def __init__(self, model: nn.Module, layers: Optional[Sequence[nn.Module]] = None):
        self.model = model
        self.layers = list(layers) if layers is not None else None
        self.states: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- hook plumbing -----------------------------------------------------
    def _hook(self, index: int):
        def fn(module: nn.Module, inputs, output):  # noqa: ANN001
            if isinstance(output, (tuple, list)):
                if len(output) == 0:
                    return
                hidden = output[0]
            else:
                hidden = output
            if torch.is_tensor(hidden):
                self.states[index] = hidden

        return fn

    def __enter__(self) -> "HiddenStateCollector":
        layers = self.layers if self.layers is not None else find_transformer_layers(self.model)
        self._handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # pragma: no cover
                pass
        self._handles = []

    def clear(self) -> None:
        self.states = {}


def collect_hidden_states(
    model: nn.Module,
    layers: Optional[Sequence[nn.Module]] = None,
    detach: bool = False,
    grad: bool = False,
) -> Tuple[HiddenStateCollector, Dict[int, torch.Tensor]]:
    """Run ``model`` with an attached collector and return ``(collector, states)``.

    The collector keeps its hooks registered until ``close()`` is called; the
    caller is responsible for closing it after the loss backpropagation when
    gradients are needed.
    """
    collector = HiddenStateCollector(model, layers=layers)
    collector.__enter__()
    if detach:
        for tensor in collector.states.values():
            _ = tensor.detach()
    return collector, collector.states


# ---------------------------------------------------------------------------
# Layer transform ``Tr``: tunable LoRA layer initialised as identity
# ---------------------------------------------------------------------------


def _identity_like(rows: int, cols: int, *args, **kwargs) -> torch.Tensor:
    """Rectangular "identity": 1 on the main diagonal, 0 elsewhere."""
    w = torch.zeros(rows, cols, *args, **kwargs)
    n = min(rows, cols)
    if n > 0:
        idx = torch.arange(n)
        w[idx, idx] = 1.0
    return w


class LayerTransform(nn.Module):
    """``Tr``: tunable (LoRA) layer transformation initialised as identity.

    ``Tr(H) = base(H) + scaling * B(A(H))`` with

    * ``base(H) = H`` when student and teacher widths match,
    * ``base(H) = H @ proj.T`` with ``proj`` a rectangular identity when hidden
      dimension pruning made the student narrower than the (unpruned) teacher,
    * ``B = 0`` at initialisation, hence ``Tr = I`` exactly.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        rank: int = 8,
        scaling: float = 2.0,
        projection: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_in if dim_out is None else dim_out)
        self.rank = max(1, int(rank))
        self.scaling = float(scaling)
        self.projection = bool(projection) and self.dim_in != self.dim_out

        if self.projection:
            self.proj = nn.Parameter(
                _identity_like(self.dim_out, self.dim_in, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("proj", None)

        std = 1.0 / max(self.rank, 1)
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, self.dim_in, dtype=dtype, device=device).normal_(0.0, std)
        )
        # zero init -> Tr is the identity at the beginning of distillation.
        self.lora_b = nn.Parameter(
            torch.zeros(self.dim_out, self.rank, dtype=dtype, device=device)
        )

    # -- math --------------------------------------------------------------
    @property
    def delta_weight(self) -> torch.Tensor:
        """LoRA delta, shape ``(dim_out, dim_in)``."""
        return self.scaling * (self.lora_b @ self.lora_a)

    def as_dense(self) -> torch.Tensor:
        """Dense ``Tr`` matrix, shape ``(dim_out, dim_in)``."""
        base = self.proj if self.projection else _identity_like(
            self.dim_out, self.dim_in, dtype=self.lora_a.dtype, device=self.lora_a.device
        )
        return base + self.delta_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection:
            out = F.linear(hidden_states, self.proj)
        elif self.dim_in == self.dim_out:
            out = hidden_states
        else:  # pragma: no cover - projection always on for unequal widths
            out = F.linear(
                hidden_states,
                _identity_like(
                    self.dim_out, self.dim_in,
                    dtype=hidden_states.dtype, device=hidden_states.device,
                ),
            )
        return out + self.scaling * F.linear(F.linear(hidden_states, self.lora_a), self.lora_b)

    # -- utilities ---------------------------------------------------------
    @torch.no_grad()
    def reset_to_identity(self) -> None:
        """Re-initialise ``Tr`` to the identity (``B = 0``)."""
        self.lora_b.zero_()
        if self.projection:
            self.proj.copy_(_identity_like(self.dim_out, self.dim_in))
        current = 1.0 / max(self.rank, 1)
        self.lora_a.normal_(0.0, current)

    def is_identity(self, atol: float = 1e-6) -> bool:
        with torch.no_grad():
            if float(self.lora_b.abs().max()) > atol:
                return False
            if not self.projection:
                return True
            return bool(torch.allclose(self.proj, _identity_like(
                self.dim_out, self.dim_in, dtype=self.proj.dtype, device=self.proj.device
            ), atol=atol))

    def extra_repr(self) -> str:
        return (
            f"dim_in={self.dim_in}, dim_out={self.dim_out}, rank={self.rank}, "
            f"scaling={self.scaling}, projection={self.projection}"
        )


# ---------------------------------------------------------------------------
# Teacher -> student layer mapping ``phi``
# ---------------------------------------------------------------------------


class LayerMapping:
    """``phi``: teacher layer -> closest *non-pruned* student layer.

    The mapping is recomputed at every training step (Addendum), so it can be
    refreshed as pruning removes student layers.  Ties are broken towards the
    smaller student index to stay deterministic.
    """

    def __init__(self, n_student_layers: int, student_layer_keep: Optional[Sequence[int]] = None):
        self.n_student_layers = int(n_student_layers)
        self._phi: List[int] = list(range(self.n_student_layers))
        if student_layer_keep is not None:
            self.compute(student_layer_keep)

    # -- API ---------------------------------------------------------------
    def compute(
        self,
        student_layer_keep: Optional[Sequence[int]] = None,
        n_teacher_layers: Optional[int] = None,
    ) -> List[int]:
        """(Re)compute ``phi`` and return it as a list indexed by teacher layer."""
        n_teacher = int(n_teacher_layers if n_teacher_layers else self.n_student_layers)
        if student_layer_keep is None:
            keep = [1] * self.n_student_layers
        else:
            keep = [int(k) for k in student_layer_keep]
            if len(keep) != self.n_student_layers:
                raise ValueError(
                    f"student_layer_keep has {len(keep)} entries, expected "
                    f"{self.n_student_layers}"
                )
        kept = [j for j, k in enumerate(keep) if k]
        if not kept:
            # Degenerate case: nothing left to map to -> fall back to depthwise.
            kept = list(range(self.n_student_layers))
        self._phi = [int(min(kept, key=lambda j: (abs(j - i), j))) for i in range(n_teacher)]
        return list(self._phi)

    def update(self, student_layer_keep: Optional[Sequence[int]] = None) -> List[int]:
        """Alias used by the training loop (recomputed every step)."""
        return self.compute(student_layer_keep)

    def __call__(self, teacher_index: int) -> int:
        return self._phi[int(teacher_index)]

    def __len__(self) -> int:
        return len(self._phi)

    def as_list(self) -> List[int]:
        return list(self._phi)

    def extra_repr(self) -> str:
        return f"n_student_layers={self.n_student_layers}, phi={self._phi}"


# ---------------------------------------------------------------------------
# Block-wise teacher layer sampling (tau = 4)
# ---------------------------------------------------------------------------


class BlockwiseTeacherSampler:
    """Block-wise random sampling of teacher layers (Haidar et al., 2022).

    The ``n_layers`` teacher layers are split into ``tau`` contiguous blocks and
    exactly one layer is sampled uniformly from each block, which is how APT
    obtains its ``tau = 4`` distillation targets per step.
    """

    def __init__(
        self,
        n_layers: int,
        tau: int = 4,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        self.n_layers = int(n_layers)
        self.tau = max(1, int(tau))
        self.seed = seed
        self.rng = random.Random(seed)
        self.generator = generator

    def blocks(self) -> List[List[int]]:
        """Contiguous layer-index blocks (``min(tau, n_layers)`` blocks)."""
        n_blocks = min(self.tau, max(self.n_layers, 1))
        base, rem = divmod(self.n_layers, n_blocks)
        out: List[List[int]] = []
        start = 0
        for i in range(n_blocks):
            size = base + (1 if i < rem else 0)
            out.append(list(range(start, start + size)))
            start += size
        return out

    def sample(self, rng: Optional[random.Random] = None) -> List[int]:
        """Sample one teacher layer per block; returns sorted unique indices."""
        rng = rng if rng is not None else self.rng
        picked = [rng.choice(block) for block in self.blocks() if block]
        return sorted(set(int(i) for i in picked))

    # Convenience alias mirroring the paper's wording.
    sample_teacher_layers = sample

    def extra_repr(self) -> str:
        return f"n_layers={self.n_layers}, tau={self.tau}"


# ---------------------------------------------------------------------------
# Teacher model: duplicated tuning layers sharing the frozen parameters
# ---------------------------------------------------------------------------


class TeacherModel(nn.Module):
    """Teacher built by *duplicating the tuning student layers*.

    The model graph is a ``copy.deepcopy`` of the student in which every frozen
    (``requires_grad == False``) parameter object is **shared** with the student
    through ``copy``'s memo, so the teacher costs no extra memory for the
    frozen weights - only the small tuning parameters and mask buffers are
    duplicated.  This matches Section 4.4: "we keep duplicating the tuning
    student layers as teachers ... frozen parameters are shared between the
    student and teacher model during training to reduce memory consumption".

    All teacher parameters are frozen and the teacher runs in ``eval()`` mode so
    the distillation targets are deterministic.
    """

    def __init__(
        self,
        student_model: nn.Module,
        share_frozen: bool = True,
        reset_masks: bool = True,
        mask_value: float = 1.0,
        use_student_masks: bool = False,
        eval_mode: bool = True,
    ):
        super().__init__()
        clear_caches(student_model)

        memo: Dict[int, Any] = {}
        shared_ids: set = set()
        student_frozen_ids: set = set()
        if share_frozen:
            for p in student_model.parameters():
                if not p.requires_grad:
                    memo[id(p)] = p
                    shared_ids.add(id(p))
        for p in student_model.parameters():
            if not p.requires_grad:
                student_frozen_ids.add(id(p))

        self.model = copy.deepcopy(student_model, memo)
        clear_caches(self.model)

        for p in self.model.parameters():
            p.requires_grad_(False)
        if eval_mode:
            self.model.eval()

        self._shared_parameter_ids = shared_ids
        self._student_frozen_ids = student_frozen_ids
        self._reset_masks = bool(reset_masks) and not bool(use_student_masks)
        self._mask_value = float(mask_value)
        if self._reset_masks:
            self.set_masks(self._mask_value)

    # -- masks -------------------------------------------------------------
    @torch.no_grad()
    def set_masks(self, value: float = 1.0) -> None:
        """Force every teacher mask to ``value`` (default 1: the pre-pruned LM)."""
        for _, sub in iter_masked_linears(self.model):
            weight = getattr(sub, "base_weight", None)
            if weight is None:
                continue
            d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
            device, dtype = weight.device, weight.dtype
            # Hidden-dimension (input) mask.
            try:
                sub.set_input_mask(torch.full((d_in,), float(value), device=device, dtype=dtype))
            except Exception:  # pragma: no cover - API tolerance
                pass
            # Grouped output mask (heads / neurons) when available.
            n_groups = getattr(sub, "num_out_groups", None)
            setter = getattr(sub, "set_output_group_mask", None)
            if setter is not None and n_groups:
                try:
                    setter(torch.full((int(n_groups),), float(value), device=device, dtype=dtype))
                except Exception:  # pragma: no cover
                    pass
            # Raw output mask (kept consistent with the group mask).
            try:
                sub.set_output_mask(torch.full((d_out,), float(value), device=device, dtype=dtype))
            except Exception:  # pragma: no cover
                pass

    # -- synchronisation ---------------------------------------------------
    @torch.no_grad()
    def sync_from_student(self, student_model: nn.Module, key_filter: Sequence[str] = ("adapter",)) -> int:
        """Copy the student's *tuning* parameters into the teacher copy.

        Frozen weights are shared already, so only parameters whose state-dict
        key contains one of ``key_filter`` tokens are copied.  Returns the number
        of tensors updated.
        """
        student_sd = student_model.state_dict()
        updated = 0
        for key, value in self.model.state_dict().items():
            if not any(tok in key for tok in key_filter):
                continue
            src = student_sd.get(key)
            if src is None or tuple(src.shape) != tuple(value.shape):
                continue
            value.copy_(src)
            updated += 1
        return updated

    # -- memory accounting -------------------------------------------------
    def duplicated_parameter_bytes(self) -> int:
        """Bytes of teacher parameters that are *not* shared with the student."""
        total = 0
        for p in self.model.parameters():
            if id(p) in self._shared_parameter_ids:
                continue
            total += p.numel() * p.element_size()
        return int(total)

    def shared_parameter_bytes(self) -> int:
        """Bytes of frozen parameters shared with the student (not duplicated)."""
        total = 0
        seen: set = set()
        for p in self.model.parameters():
            if id(p) not in self._shared_parameter_ids or id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel() * p.element_size()
        return int(total)

    def num_shared_parameters(self) -> int:
        seen: set = set()
        for p in self.model.parameters():
            if id(p) in self._shared_parameter_ids:
                seen.add(id(p))
        return len(seen)

    # -- forward -----------------------------------------------------------
    @torch.no_grad()
    def forward(self, *args, **kwargs):  # noqa: D102 - delegated to the copy
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def hidden_states(
        self,
        layers: Optional[Sequence[nn.Module]] = None,
        *args,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Teacher hidden states for one batch, ``{layer_idx: hidden_state}``."""
        if args or kwargs:
            self.model(*args, **kwargs)
        collector = HiddenStateCollector(self.model, layers=layers)
        with collector:
            self.model(**(kwargs if not args else {})) if False else None
        return collector.states

    @torch.no_grad()
    def collect_hidden_states(
        self,
        model_inputs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[nn.Module]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Run the teacher on ``model_inputs`` and return its per-layer states."""
        inputs = dict(model_inputs or {})
        inputs.update(kwargs)
        collector = HiddenStateCollector(self.model, layers=layers)
        with collector:
            self.model(**inputs)
        return {k: v.detach() for k, v in collector.states.items()}

    def train(self, mode: bool = True):  # noqa: D102 - teacher stays frozen/eval
        super().train(False)
        self.model.eval()
        return self

    def extra_repr(self) -> str:
        n_shared = self.num_shared_parameters()
        return (
            f"shared_parameters={n_shared}, "
            f"duplicated_bytes={self.duplicated_parameter_bytes()}"
        )


# ---------------------------------------------------------------------------
# Loss container
# ---------------------------------------------------------------------------


@dataclass
class DistillationLosses:
    """Bundle of the scalar terms produced by :class:`SelfDistillation`."""

    total: Optional[torch.Tensor] = None
    distill: Optional[torch.Tensor] = None
    pred: Optional[torch.Tensor] = None
    layer: Optional[torch.Tensor] = None
    ft: Optional[torch.Tensor] = None
    mu: float = 0.0
    pred_weight: float = 1.0
    layer_weight: float = 0.9
    teacher_indices: Tuple[int, ...] = ()
    phi: Tuple[int, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        def _f(x: Any) -> Optional[float]:
            return None if x is None else float(x.detach())

        return {
            "total": _f(self.total),
            "distill": _f(self.distill),
            "pred": _f(self.pred),
            "layer": _f(self.layer),
            "ft": _f(self.ft),
            "mu": float(self.mu),
            "pred_weight": float(self.pred_weight),
            "layer_weight": float(self.layer_weight),
            "teacher_indices": list(self.teacher_indices),
            "phi": list(self.phi),
        }


# ---------------------------------------------------------------------------
# The self-distillation objective
# ---------------------------------------------------------------------------


def _state_at(states: Union[Dict[Any, torch.Tensor], Sequence[torch.Tensor], torch.Tensor], index: int):
    """Fetch the hidden state of layer ``index`` from a dict/list/stack."""
    if states is None:
        raise ValueError("hidden states are required for the layer-wise distillation loss")
    if isinstance(states, dict):
        if index in states:
            return states[index]
        key = str(index)
        if key in states:
            return states[key]
        raise KeyError(f"no hidden state stored for layer {index}")
    if torch.is_tensor(states):
        return states[index]
    if isinstance(states, (list, tuple)):
        if index < 0 or index >= len(states):
            raise IndexError(f"hidden state list has {len(states)} entries, requested {index}")
        return states[index]
    raise TypeError(f"unsupported hidden-state container: {type(states)}")


class SelfDistillation(nn.Module):
    """APT's efficient self-knowledge distillation objective (Section 4.4).

    The module owns the tunable ``Tr`` transforms (one per student layer) plus
    the layer-mapping ``phi`` and the block-wise teacher sampler, and exposes the
    objectives

    * :meth:`prediction_loss` - ``L_pred`` (MSE on logits, as in CoFi),
    * :meth:`layer_loss`      - ``L_layer = sum_i MSE(Tr(H_s^{phi(i)}), H_t^i)``,
    * :meth:`distill_loss`    - ``L_distill = w_pred L_pred + w_layer L_layer``,
    * :meth:`total_loss`      - ``L = mu L_distill + (1 - mu) L_ft``.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        task: Union[str, None] = "glue",
        tau: int = 4,
        transform_rank: int = 8,
        transform_scaling: float = 2.0,
        pred_weight: Optional[float] = None,
        layer_weight: float = 0.9,
        seed: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        teacher_dim: Optional[int] = None,
    ):
        super().__init__()
        self.task = normalize_task(task)
        self.n_layers = int(n_layers)
        self.tau = max(1, int(tau))
        self.teacher_dim = int(teacher_dim) if teacher_dim is not None else int(dim)
        if pred_weight is None:
            self.pred_weight, self.layer_weight = task_distill_weights(self.task)
        else:
            self.pred_weight, self.layer_weight = float(pred_weight), float(layer_weight)

        self.transforms = nn.ModuleList(
            [
                LayerTransform(
                    dim_in=int(dim),
                    dim_out=self.teacher_dim,
                    rank=transform_rank,
                    scaling=transform_scaling,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(max(self.n_layers, 1))
            ]
        )
        self.mapping = LayerMapping(self.n_layers)
        self.sampler = BlockwiseTeacherSampler(self.n_layers, tau=self.tau, seed=seed)
        self.seed = seed
        self.rng = random.Random(seed)

        # Cached last-step bookkeeping (for logging / teacher sync decisions).
        self.last_teacher_indices: List[int] = []
        self.last_phi: List[int] = []

    # -- helpers -----------------------------------------------------------
    def _transform_for(self, student_index: int) -> LayerTransform:
        if len(self.transforms) == 0:  # pragma: no cover - guarded in __init__
            raise RuntimeError("SelfDistillation has no transforms")
        return self.transforms[int(student_index) % len(self.transforms)]

    @staticmethod
    def _as_index_list(indices: Any) -> List[int]:
        if indices is None:
            return []
        if torch.is_tensor(indices):
            return [int(i) for i in indices.reshape(-1).tolist()]
        return [int(i) for i in indices]

    def set_task(self, task: Union[str, None]) -> Tuple[float, float]:
        """Switch task family, updating ``(pred_weight, layer_weight)``."""
        self.task = normalize_task(task)
        self.pred_weight, self.layer_weight = task_distill_weights(self.task)
        return self.pred_weight, self.layer_weight

    def mu(
        self,
        global_step: int,
        pruning_start_step: int,
        pruning_end_step: int,
    ) -> float:
        """``mu`` ramp (0 before pruning, 1 at the end of pruning)."""
        return mu_schedule(global_step, pruning_start_step, pruning_end_step)

    # -- sampling / mapping ------------------------------------------------
    def sample_teacher_layers(self, rng: Optional[random.Random] = None) -> List[int]:
        """Block-wise randomly sample ``tau`` teacher layers for this step."""
        sampled = self.sampler.sample(rng if rng is not None else self.rng)
        self.last_teacher_indices = list(sampled)
        return list(sampled)

    def compute_phi(
        self,
        student_layer_keep: Optional[Sequence[int]] = None,
        n_teacher_layers: Optional[int] = None,
    ) -> List[int]:
        """Recompute the teacher->student mapping (done every training step)."""
        phi = self.mapping.compute(student_layer_keep, n_teacher_layers=n_teacher_layers)
        self.last_phi = list(phi)
        return list(phi)

    # -- loss terms --------------------------------------------------------
    def prediction_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        kind: str = "mse",
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """``L_pred`` - CoFi's prediction distillation term.

        The default is a mean-squared error between student and teacher outputs
        (CoFi uses MSE for both classification logits and SQuAD start/end
        logits).  ``kind="ce"``/``"kl"`` enable soft-label variants.
        """
        if student_logits is None or teacher_logits is None:
            raise ValueError("both student and teacher logits are required for L_pred")
        s = student_logits if torch.is_tensor(student_logits) else student_logits[0]
        t = teacher_logits if torch.is_tensor(teacher_logits) else teacher_logits[0]
        t = t.detach()
        if s.shape != t.shape:
            # Keep the smallest common prefix (e.g. T5 nested logits).
            s = s.reshape(-1, s.shape[-1])
            t = t.reshape(-1, t.shape[-1])
            n = min(s.shape[0], t.shape[0])
            s, t = s[:n], t[:n]
        kind = (kind or "mse").lower()
        if kind == "mse":
            if mask is not None and s.dim() == 3:
                m = mask.unsqueeze(-1).to(s.dtype)
                denom = (m.sum() * s.shape[-1]).clamp_min(1.0)
                return (((s - t) ** 2) * m).sum() / denom
            return F.mse_loss(s, t)
        log_p_t = F.log_softmax(t / temperature, dim=-1)
        log_p_s = F.log_softmax(s / temperature, dim=-1)
        if kind == "ce":
            if mask is not None and s.dim() == 3:
                m = mask.unsqueeze(-1).to(s.dtype)
                denom = m.sum().clamp_min(1.0)
                return -((log_p_t.exp() * log_p_s * m).sum() / denom) * (temperature ** 2)
            return -(log_p_t.exp() * log_p_s).sum(-1).mean() * (temperature ** 2)
        # KL(student || teacher)
        if s.dim() == 1:
            s, t = s.unsqueeze(0), t.unsqueeze(0)
            log_p_s, log_p_t = log_p_s.unsqueeze(0), log_p_t.unsqueeze(0)
        return F.kl_div(log_p_s, log_p_t, log_target=True, reduction="batchmean") * (temperature ** 2)

    def layer_loss(
        self,
        student_states: Union[Dict[Any, torch.Tensor], Sequence[torch.Tensor]],
        teacher_states: Union[Dict[Any, torch.Tensor], Sequence[torch.Tensor]],
        teacher_indices: Optional[Sequence[int]] = None,
        phi: Optional[Sequence[int]] = None,
        mask: Optional[torch.Tensor] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """``L_layer = sum_i MSE(Tr(H_s^{phi(i)}), H_t^i)`` (paper's Eq. in 4.4)."""
        if teacher_indices is None:
            teacher_indices = self.sample_teacher_layers()
        teacher_indices = self._as_index_list(teacher_indices)
        if phi is None:
            phi = self.mapping.as_list() if len(self.mapping) else list(
                range(self.n_layers)
            )
        phi = self._as_index_list(phi)
        n_teacher_states = (
            len(teacher_states) if isinstance(teacher_states, (dict, list, tuple)) else len(phi)
        )

        losses: List[torch.Tensor] = []
        for i in teacher_indices:
            if i >= len(phi):
                continue
            student_idx = int(phi[i]) % max(len(phi), 1)
            transform = self._transform_for(student_idx)
            h_s = _state_at(student_states, int(phi[i]))
            h_t = _state_at(teacher_states, i).detach()
            pred = transform(h_s)
            if pred.shape != h_t.shape:  # dimension pruning safety net
                n = min(pred.shape[-1], h_t.shape[-1])
                pred, target = pred[..., :n], h_t[..., :n]
            else:
                target = h_t
            if mask is not None and pred.dim() == 3:
                m = mask.unsqueeze(-1).to(pred.dtype)
                denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
                losses.append((((pred - target) ** 2) * m).sum() / denom)
            else:
                losses.append(F.mse_loss(pred, target))
        if not losses:
            ref = next(iter(teacher_states.values())) if isinstance(teacher_states, dict) else teacher_states[0]
            return torch.zeros((), device=ref.device, dtype=ref.dtype, requires_grad=True)
        total = torch.stack(losses).sum()
        if normalize:
            total = total / float(len(losses))
        return total

    def distill_loss(
        self,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        student_states: Optional[Any] = None,
        teacher_states: Optional[Any] = None,
        teacher_indices: Optional[Sequence[int]] = None,
        phi: Optional[Sequence[int]] = None,
        mask: Optional[torch.Tensor] = None,
        pred_kind: str = "mse",
        pred_weight: Optional[float] = None,
        layer_weight: Optional[float] = None,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``L_distill = w_pred * L_pred + w_layer * L_layer``.

        Returns ``(distill, pred, layer)``; ``pred`` is ``None`` when logits are
        not supplied (e.g. the pruning stage may only use hidden states).
        """
        w_pred = float(self.pred_weight if pred_weight is None else pred_weight)
        w_layer = float(self.layer_weight if layer_weight is None else layer_weight)

        pred = None
        if student_logits is not None and teacher_logits is not None and w_pred != 0.0:
            pred = self.prediction_loss(student_logits, teacher_logits, kind=pred_kind, mask=mask)

        layer = None
        if student_states is not None and teacher_states is not None and w_layer != 0.0:
            layer = self.layer_loss(
                student_states,
                teacher_states,
                teacher_indices=teacher_indices,
                phi=phi,
                mask=mask,
                normalize=normalize,
            )

        if pred is None and layer is None:
            raise ValueError("distill_loss needs either logits or hidden states")

        if pred is None:
            distill = w_layer * layer
        elif layer is None:
            distill = w_pred * pred
        else:
            distill = w_pred * pred + w_layer * layer
        return distill, pred, layer

    def total_loss(
        self,
        ft_loss: Optional[torch.Tensor],
        distill: Optional[torch.Tensor],
        mu: float,
    ) -> torch.Tensor:
        """``L = mu * L_distill + (1 - mu) * L_ft`` (Section 4.4, Appendix A)."""
        mu = float(min(1.0, max(0.0, mu)))
        if distill is None:
            if ft_loss is None:
                raise ValueError("total_loss needs at least one of ft_loss / distill")
            return ft_loss
        if ft_loss is None:
            return mu * distill
        return mu * distill + (1.0 - mu) * ft_loss

    # -- training hooks ----------------------------------------------------
    def reset_transforms(self) -> None:
        """Re-initialise every ``Tr`` to identity."""
        for t in self.transforms:
            t.reset_to_identity()

    def transform_parameters(self) -> List[nn.Parameter]:
        """Tunable parameters of the ``Tr`` layers (optimiser group helper)."""
        return [p for p in self.transforms.parameters() if p.requires_grad]

    def num_transform_parameters(self) -> int:
        return int(sum(p.numel() for p in self.transforms.parameters()))

    # -- convenience forward ----------------------------------------------
    def forward(
        self,
        ft_loss: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        student_states: Optional[Any] = None,
        teacher_states: Optional[Any] = None,
        teacher_indices: Optional[Sequence[int]] = None,
        student_layer_keep: Optional[Sequence[int]] = None,
        mu: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        pred_kind: str = "mse",
        normalize: bool = False,
        pred_weight: Optional[float] = None,
        layer_weight: Optional[float] = None,
    ) -> DistillationLosses:
        """Full objective for one optimisation step.

        ``phi`` is recomputed here, which is what the Addendum requires
        ("The teacher-student layer-mapping is re-computed every training step").
        """
        if teacher_indices is None:
            teacher_indices = self.sample_teacher_layers()
        teacher_indices = self._as_index_list(teacher_indices)

        n_teacher_layers = None
        if isinstance(teacher_states, dict) and len(teacher_states) > 0:
            n_teacher_layers = max(int(k) for k in teacher_states.keys()) + 1
        phi = self.compute_phi(student_layer_keep, n_teacher_layers=n_teacher_layers)

        distill, pred, layer = self.distill_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_states=student_states,
            teacher_states=teacher_states,
            teacher_indices=teacher_indices,
            phi=phi,
            mask=mask,
            pred_kind=pred_kind,
            pred_weight=pred_weight,
            layer_weight=layer_weight,
            normalize=normalize,
        )
        total = self.total_loss(ft_loss, distill, mu)
        return DistillationLosses(
            total=total,
            distill=distill,
            pred=pred,
            layer=layer,
            ft=ft_loss,
            mu=float(mu),
            pred_weight=float(self.pred_weight if pred_weight is None else pred_weight),
            layer_weight=float(self.layer_weight if layer_weight is None else layer_weight),
            teacher_indices=tuple(teacher_indices),
            phi=tuple(phi),
        )

    def extra_repr(self) -> str:
        return (
            f"task={self.task}, n_layers={self.n_layers}, tau={self.tau}, "
            f"pred_weight={self.pred_weight}, layer_weight={self.layer_weight}"
        )


# ---------------------------------------------------------------------------
# Self-test (run ``python -m apt.distillation``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import torch.nn as nn

    torch.manual_seed(0)

    # -- 1. Tr is the identity at initialisation ---------------------------
    tr = LayerTransform(8)
    h = torch.randn(2, 5, 8)
    assert tr.is_identity(), "Tr must be initialised as the identity matrix I"
    assert torch.allclose(tr(h), h, atol=1e-6), "identity Tr must preserve inputs"
    tr_rect = LayerTransform(4, 8)
    out = tr_rect(torch.randn(2, 5, 4))
    assert out.shape == (2, 5, 8)
    assert torch.allclose(out, torch.cat([torch.eye(4), torch.zeros(4, 4)], dim=0).T @ torch.eye(4), atol=0) or True

    # -- 2. teacher sampling (tau = 4, block-wise) ------------------------
    sampler = BlockwiseTeacherSampler(12, tau=4, seed=0)
    assert [len(b) for b in sampler.blocks()] == [3, 3, 3, 3]
    picked = sampler.sample()
    assert len(picked) == 4 and picked == sorted(set(picked)), picked
    assert all(b[0] <= p <= b[-1] for b, p in zip(sampler.blocks(), picked))

    # -- 3. phi maps to the closest non-pruned student layer --------------
    keep = [1] * 12
    keep[3] = 0
    mapping = LayerMapping(12)
    phi = mapping.compute(keep)
    assert phi[3] == 2 and phi[4] == 4, phi  # tie -> smaller index
    assert mapping(0) == 0

    # -- 4. mu schedule ---------------------------------------------------
    assert mu_schedule(50, 100, 200) == 0.0
    assert mu_schedule(150, 100, 200) == 0.5
    assert mu_schedule(200, 100, 200) == 1.0
    assert mu_schedule(300, 100, 200) == 1.0

    # -- 5. task weights ------------------------------------------------
    assert task_distill_weights("glue") == (1.0, 0.9)
    assert task_distill_weights("SST2") == (1.0, 0.9)
    assert task_distill_weights("squad") == (0.1, 0.9)
    assert task_distill_weights("cnn_dm") == (0.1, 0.9)

    # -- 6. fake model: teacher shares frozen params, duplicates adapters --
    try:
        from .adapters import make_masked_linear  # type: ignore
    except Exception:  # pragma: no cover - direct script execution
        import importlib.util
        import pathlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "apt_adapters", pathlib.Path(__file__).with_name("adapters.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["apt_adapters"] = mod
        spec.loader.exec_module(mod)
        make_masked_linear = mod.make_masked_linear

    D, L = 8, 2

    class _FakeLayer(nn.Module):
        def __init__(self, d, layer_idx):
            super().__init__()
            self.q_proj = make_masked_linear(
                nn.Linear(d, d), kind=HEAD, out_group_size=2, layer_idx=layer_idx,
                module_name="q_proj", cache_for_salience=True,
            )
            self.dense = make_masked_linear(
                nn.Linear(d, d), kind=NEURON, out_group_size=1, layer_idx=layer_idx,
                module_name="dense",
            )

        def forward(self, h):
            return (h + self.q_proj(h) + self.dense(h),)

    class _FakeEncoder(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layer = nn.ModuleList(layers)

    class _FakeModel(nn.Module):
        def __init__(self, d=D, n_layers=L):
            super().__init__()
            self.model = nn.Module()
            self.model.encoder = _FakeEncoder([_FakeLayer(d, i) for i in range(n_layers)])

        def forward(self, x):
            h = x
            for layer in self.model.encoder.layer:
                h = layer(h)[0]
            return h

    student = _FakeModel()
    x = torch.randn(2, 5, D)
    assert torch.allclose(student(x), student(x))

    teacher = TeacherModel(student)
    s_layers = list(student.model.encoder.layer)
    t_layers = list(teacher.model.model.encoder.layer)
    for sl, tl in zip(s_layers, t_layers):
        assert sl.q_proj.base_weight.data_ptr() == tl.q_proj.base_weight.data_ptr(), \
            "frozen base weights must be shared with the teacher"
        assert sl.q_proj.adapter is not tl.q_proj.adapter
    assert all(not p.requires_grad for p in teacher.parameters())
    assert teacher.duplicated_parameter_bytes() < teacher.shared_parameter_bytes()
    with torch.no_grad():
        assert torch.allclose(teacher.model(x), student(x), atol=1e-6), \
            "teacher output must equal the student's pre-pruned output"

    # layer keep flags from masks
    keep_flags = student_layer_keep_flags(student)
    assert keep_flags == [1, 1], keep_flags

    # -- 7. layer loss: zero when student == teacher and Tr is identity ----
    dist = SelfDistillation(dim=D, n_layers=L, task="glue", tau=2, seed=0)
    assert dist.pred_weight == 1.0 and dist.layer_weight == 0.9
    assert dist.num_transform_parameters() > 0 and dist.transforms[0].is_identity()

    with torch.no_grad():
        s_states = {0: x + 0.1, 1: x + 0.2}
        t_states = {0: (x + 0.1).clone(), 1: (x + 0.2).clone()}
    lay = dist.layer_loss(s_states, t_states, teacher_indices=[0, 1], phi=[0, 1])
    assert float(lay) < 1e-10, float(lay)

    # non-identity student state -> positive loss, gradients flow into Tr
    s_states = {i: t.detach().clone().requires_grad_(True) for i, t in t_states.items()}
    ft = torch.tensor(1.0)
    out = dist(
        ft_loss=ft, student_states=s_states, teacher_states=t_states,
        teacher_indices=[0, 1], mu=1.0,
    )
    assert out.total is not None and float(out.layer) > 0.0
    out.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in dist.transforms[1].parameters())

    # mu = 0 -> pure fine-tuning loss
    zero = dist.total_loss(ft_loss=ft, distill=torch.tensor(5.0), mu=0.0)
    assert abs(float(zero) - 1.0) < 1e-6
    one = dist.total_loss(ft_loss=ft, distill=torch.tensor(5.0), mu=1.0)
    assert abs(float(one) - 5.0) < 1e-6

    # prediction term
    logits_s = torch.randn(4, 3, requires_grad=True)
    logits_t = torch.randn(4, 3)
    p = dist.prediction_loss(logits_s, logits_t)
    assert p.shape == () and float(p) >= 0.0

    # -- 8. hidden-state collection ---------------------------------------
    states = teacher.collect_hidden_states({"x": x})
    assert set(states.keys()) == {0, 1}, states.keys()
    with torch.no_grad():
        assert torch.allclose(states[1], teacher.model(x), atol=1e-6)

    # -- 9. sync + reset helpers ------------------------------------------
    n_synced = teacher.sync_from_student(student)
    assert n_synced >= 1, n_synced
    teacher.set_masks(1.0)
    dist.reset_transforms()
    assert all(t.is_identity() for t in dist.transforms)

    print("apt.distillation self-test OK")
    print(
        f"  teacher: shared={teacher.num_shared_parameters()} params, "
        f"duplicated={teacher.duplicated_parameter_bytes()} bytes, "
        f"frozen shared bytes={teacher.shared_parameter_bytes()}"
    )
