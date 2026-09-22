"""APT adapters: LoRA-style adapters with binary input/output pruning masks.

Implements the APT adapter described in Section 4.1 of *APT: Adaptive Pruning and
Tuning Pretrained Language Models for Efficient Training and Inference*:

    H_apt(X) = m_o  ∘  (W + s · W_B W_A) X  ∘  m_i            (Eq. 1)

with

    * ``W``            frozen pretrained weight (``d_o x d_i``),
    * ``W_A``          tuning parameter (``r_apt x d_i``), Gaussian init N(0, sigma^2),
    * ``W_B``          tuning parameter (``d_o x r_apt``), zero init (LoRA init),
    * ``s``            constant scaling factor (set statically to 2 in the paper, App. A),
    * ``m_i`` (``d_i``) input mask -- prunes the transformer hidden dimension,
    * ``m_o`` (``d_o``) output mask -- prunes MHA heads / FFN internal neurons,
    * ``r_apt``        dynamic adapter rank (starts at 8, App. A).

The masks are *binary* pruning masks during inference ("The parameter block is pruned
when the multiplying mask is set to 0 and retained when set to 1", Sec. 4.1, App. C)
but they are annealed towards their binary value during training: "we gradually
decrease the pruning masks of pruned blocks by alpha < 1 instead of instantly setting
them from ones to zeros" with ``alpha = 0.01`` (App. A / App. C).  Hence masks are
kept as float tensors in ``[0, 1]`` and multiplied with the activations.

Rank growth (Section 4.3): when the tuning budget increases from Delta_t to
Delta_t', salient adapters get ``r_apt' = floor(r_apt * Delta_t' / Delta_t)``
(Equation (3') in Section 4.3) and, for training stability, the extra rows of ``W_A``
are random Gaussian ``N(0, sigma^2)`` and the extra columns of ``W_B`` are zeros
"so the layer's output remains unchanged before and after new parameters added".

After training, ``m_o ∘ (W + s W_B W_A) ∘ m_i`` can be merged into a dense weight and
the pruned heads / neurons / hidden dimensions physically removed (see ``apt.merge``);
the tuning parameters therefore add no inference overhead (Eq. 1 footnote 2, Sec. 6).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "APTAdapter",
    "MaskedLinear",
    "make_masked_linear",
    "HEAD",
    "NEURON",
    "DIMENSION",
    "BLOCK_TYPES",
]

# Block-type identifiers, matching the block category function f(b) of Appendix C:
#   f(b_i) = 0 if head, 1 if neuron, 2 if dimension
HEAD = 0
NEURON = 1
DIMENSION = 2
BLOCK_TYPES = {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}


def _lora_default_std(rank: int) -> float:
    """Default Gaussian std for ``W_A`` rows.

    The paper only states ``N(0, sigma^2)`` ("concatenate random Gaussian initialized
    parameters N(0, sigma^2) in W_A and zeros in W_B same as the LoRA initialization",
    Sec. 4.3).  We follow the common LoRA Gaussian initialisation, ``std = 1 / r``,
    which is also what PEFT uses for ``init_lora_weights="gaussian"``.  Since ``W_B``
    is zero-initialised, the value of sigma does not change the layer output.
    """
    return 1.0 / float(max(rank, 1))


class APTAdapter(nn.Module):
    """Low-rank tuning parameters ``W_B W_A`` of an APT adapter (Eq. 1).

    The adapter owns only the tuning parameters; the frozen weight, the masks and the
    ``m_o ∘ (·) ∘ m_i`` sandwiching live in :class:`MaskedLinear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        scaling: float = 2.0,
        w_a_std: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._rank = int(rank)
        self.scaling = float(scaling)
        self.w_a_std = float(_lora_default_std(rank) if w_a_std is None else w_a_std)

        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight_a = nn.Parameter(
            torch.empty(self._rank, self.in_features, **factory_kwargs)
        )
        self.weight_b = nn.Parameter(
            torch.zeros(self.out_features, self._rank, **factory_kwargs)
        )
        self.reset_weight_a()

    # ------------------------------------------------------------------ helpers
    def reset_weight_a(self) -> None:
        """Gaussian init of ``W_A`` (``W_B`` stays zero => output unchanged)."""
        with torch.no_grad():
            self.weight_a.normal_(mean=0.0, std=self.w_a_std)

    @property
    def rank(self) -> int:
        """Current ``r_apt``."""
        return self._rank

    @property
    def device(self) -> torch.device:
        return self.weight_a.device

    def num_tuning_parameters(self) -> int:
        """Number of tuning parameters ``delta(Theta_t, M_t, R_t)`` in this adapter."""
        return int(self.weight_a.numel() + self.weight_b.numel())

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``s * W_B (W_A x)`` (the LoRA term of Eq. 1)."""
        if self._rank == 0:
            return torch.zeros(
                *x.shape[:-1], self.out_features, dtype=x.dtype, device=x.device
            )
        h = F.linear(x, self.weight_a)  # (..., r_apt)
        h = F.linear(h, self.weight_b)  # (..., d_o)
        if self.scaling != 1.0:
            h = h * self.scaling
        return h

    # ------------------------------------------------------- dynamic rank growth
    @torch.no_grad()
    def increase_rank(self, new_rank: int) -> bool:
        """Grow ``r_apt`` to ``new_rank`` preserving the layer output.

        New ``W_A`` rows are Gaussian ``N(0, sigma^2)`` and new ``W_B`` columns are
        zeros (Sec. 4.3, LoRA initialisation), so ``W_B W_A`` is numerically unchanged.
        """
        new_rank = int(new_rank)
        if new_rank <= self._rank:
            return False
        extra = new_rank - self._rank
        device, dtype = self.weight_a.device, self.weight_a.dtype
        new_a = torch.empty(extra, self.in_features, dtype=dtype, device=device)
        new_a.normal_(mean=0.0, std=self.w_a_std)
        new_b = torch.zeros(self.out_features, extra, dtype=dtype, device=device)
        self.weight_a = nn.Parameter(torch.cat([self.weight_a.detach(), new_a], dim=0))
        self.weight_b = nn.Parameter(torch.cat([self.weight_b.detach(), new_b], dim=1))
        self._rank = new_rank
        return True

    @torch.no_grad()
    def set_rank(self, new_rank: int) -> bool:
        """Set ``r_apt`` to ``new_rank`` (only growth is supported, per Sec. 4.3)."""
        new_rank = int(new_rank)
        if new_rank == self._rank:
            return False
        if new_rank < self._rank:
            raise ValueError(
                "APT only increases adapter ranks during training "
                f"(requested {new_rank} < current {self._rank})"
            )
        return self.increase_rank(new_rank)

    # -------------------------------------------------------------- bookkeeping
    def importance(self) -> torch.Tensor:
        """``I(H_apt) = sum_{i,j} S(W_B,ij)`` (Sec. 4.3).

        ``S`` is the magnitude of the weight-gradient product, Eq. (2); the caller
        supplies salience through :meth:`accumulate_salience`, so this returns the
        cached value when available and otherwise falls back to ``|W_B|``.
        """
        if self._b_salience is not None:
            return self._b_salience.sum()
        return self.weight_b.detach().abs().sum()

    _b_salience: Optional[torch.Tensor] = None

    @torch.no_grad()
    def set_b_salience(self, salience: Optional[torch.Tensor]) -> None:
        """Store per-entry salience of ``W_B`` used by :meth:`importance`."""
        self._b_salience = None if salience is None else salience.detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self._rank}, scaling={self.scaling}"
        )


class MaskedLinear(nn.Module):
    """Frozen ``nn.Linear`` + :class:`APTAdapter` with input/output pruning masks.

    Computes Eq. (1)::

        H = m_o ∘ ( W x' + s · W_B (W_A x') )    with x' = m_i ∘ x

    ``m_i`` (``d_i``) prunes the transformer hidden dimension, ``m_o`` (``d_o``)
    prunes MHA heads and FFN internal neurons (Sec. 4.1).  ``m_o`` may be *grouped*:
    a single mask value covers ``out_group_size`` contiguous output units (``d_h`` for
    an attention head, ``1`` for a single FFN neuron).

    The frozen weight lives in ``base_weight`` (a non-trainable ``nn.Parameter`` so it
    survives ``state_dict`` round-trips and can be replaced at merge time); the base
    bias, if any, is frozen as well.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter: Optional[APTAdapter] = None,
        mask_in: Optional[torch.Tensor] = None,
        mask_out: Optional[torch.Tensor] = None,
        kind: int = HEAD,
        out_group_size: int = 1,
        layer_idx: int = -1,
        module_name: str = "",
        cache_for_salience: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be an nn.Linear")
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.kind = int(kind)
        self.layer_idx = int(layer_idx)
        self.module_name = str(module_name)

        self.base_weight = nn.Parameter(
            base_layer.weight.detach().clone(), requires_grad=False
        )
        if base_layer.bias is not None:
            self.base_bias = nn.Parameter(
                base_layer.bias.detach().clone(), requires_grad=False
            )
        else:
            self.register_parameter("base_bias", None)

        if out_group_size <= 0:
            raise ValueError("out_group_size must be positive")
        if self.out_features % out_group_size != 0:
            raise ValueError(
                f"out_features ({self.out_features}) not divisible by "
                f"out_group_size ({out_group_size})"
            )
        self.out_group_size = int(out_group_size)

        self.adapter = adapter

        # masks: float tensors in [0, 1]; 1 => retained, 0 => pruned (Sec. 4.1, App. C)
        if mask_in is None:
            mask_in = torch.ones(self.in_features)
        else:
            mask_in = mask_in.detach().clone()
        if mask_out is None:
            mask_out = torch.ones(self.out_features)
        else:
            mask_out = mask_out.detach().clone()
        self.register_buffer("mask_in", mask_in.to(torch.float32).reshape(-1))
        self.register_buffer("mask_out", mask_out.to(torch.float32).reshape(-1))

        # activation / gradient caching for salience scoring (Eq. 5, App. B)
        self.cache_for_salience = bool(cache_for_salience)
        self.register_buffer("_cached_input", None, persistent=False)
        self.register_buffer("_cached_output", None, persistent=False)
        self.register_buffer("_cached_output_grad", None, persistent=False)
        self._capture_grad = False
        self._logical_output_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ devices
    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        """Keep cached activation hooks consistent after ``.to()`` / ``.half()``."""
        super()._apply(fn, recurse=recurse)
        self._cached_input = None
        self._cached_output = None
        self._cached_output_grad = None
        return self

    # ------------------------------------------------------------------- shapes
    @property
    def num_out_groups(self) -> int:
        """Number of prunable output blocks (heads / neurons)."""
        return self.out_features // self.out_group_size

    @property
    def rank(self) -> int:
        return 0 if self.adapter is None else self.adapter.rank

    def num_tuning_parameters(self) -> int:
        return 0 if self.adapter is None else self.adapter.num_tuning_parameters()

    # -------------------------------------------------------------------- masks
    @torch.no_grad()
    def set_input_mask(self, values: torch.Tensor) -> None:
        """Set ``m_i`` (``d_i`` values in ``[0, 1]``).

        If the wrapper passes the *same* tensor object to several modules, the buffer
        is shared and in-place updates propagate to all of them (used for the hidden
        dimension mask of the residual stream).
        """
        values = values.to(self.mask_in.device, torch.float32).reshape(-1)
        if values.numel() != self.in_features:
            raise ValueError(
                f"mask_in must have {self.in_features} entries, got {values.numel()}"
            )
        if values.data_ptr() == self.mask_in.data_ptr():
            return
        self.mask_in = values

    @torch.no_grad()
    def set_output_mask(self, values: torch.Tensor) -> None:
        """Set ``m_o`` given per-output-unit values (``d_o``)."""
        values = values.to(self.mask_out.device, torch.float32).reshape(-1)
        if values.numel() != self.out_features:
            raise ValueError(
                f"mask_out must have {self.out_features} entries, got {values.numel()}"
            )
        self.mask_out = values
        self._logical_output_mask = values.detach().clone()

    @torch.no_grad()
    def set_output_group_mask(self, values: torch.Tensor) -> None:
        """Set ``m_o`` from per-block (head / neuron) values, expanded to ``d_o``."""
        values = values.to(self.mask_out.device, torch.float32).reshape(-1)
        if values.numel() != self.num_out_groups:
            raise ValueError(
                f"group mask must have {self.num_out_groups} entries, got {values.numel()}"
            )
        self._logical_output_mask = values.detach().clone()
        expanded = values.repeat_interleave(self.out_group_size)
        if expanded.numel() != self.out_features:  # pragma: no cover - defensive
            expanded = expanded[: self.out_features]
        self.mask_out = expanded

    @torch.no_grad()
    def get_output_group_mask(self) -> torch.Tensor:
        """Return per-block (head / neuron) values implied by ``m_o``."""
        grouped = self.mask_out.reshape(self.num_out_groups, self.out_group_size)
        return grouped.mean(dim=1)

    def prune_count(self) -> int:
        """Number of pruned blocks for this module (heads / neurons / dims)."""
        return int((self.get_output_group_mask() <= 0.5).sum().item())

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``m_o ∘ (W + s W_B W_A) (m_i ∘ x)`` -- Eq. (1)."""
        x_in = x
        if self.mask_in is not None:
            x_in = x * self.mask_in.to(x.dtype)

        y = F.linear(x_in, self.base_weight, self.base_bias)
        if self.adapter is not None and self.adapter.rank > 0:
            y = y + self.adapter(x_in).to(y.dtype)
        if self.mask_out is not None:
            y = y * self.mask_out.to(y.dtype)

        if self.cache_for_salience:
            self._cached_input = x_in.detach()
            if self._capture_grad:
                self._cached_output_grad = None

                def _save_grad(grad: torch.Tensor) -> None:
                    self._cached_output_grad = grad.detach()

                y.register_hook(_save_grad)
            self._cached_output = y
        return y

    @torch.no_grad()
    def merged_weight(self, threshold: float = 0.0) -> torch.Tensor:
        """Return ``m_o ∘ (W + s W_B W_A) ∘ m_i`` as a dense weight (Sec. 4.1 fn. 2).

        ``threshold`` optionally hardens the annealed masks before merging (values
        ``<= threshold`` count as pruned).
        """
        m_in = self.mask_in
        m_out = self.mask_out
        if threshold > 0.0:
            m_in = (m_in > threshold).to(m_in.dtype)
            m_out = (m_out > threshold).to(m_out.dtype)
        weight = self.base_weight.detach() * m_in.unsqueeze(0)
        if self.adapter is not None and self.adapter.rank > 0:
            w_b = self.adapter.weight_b.detach() * m_out.unsqueeze(1)
            w_a = self.adapter.weight_a.detach() * m_in.unsqueeze(0)
            weight = weight + self.adapter.scaling * (w_b @ w_a)
        weight = weight * m_out.unsqueeze(1)
        return weight

    def extra_repr(self) -> str:
        kind = BLOCK_TYPES.get(self.kind, str(self.kind))
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"kind={kind}, out_group_size={self.out_group_size}, rank={self.rank}, "
            f"layer_idx={self.layer_idx}, module={self.module_name!r}"
        )


def make_masked_linear(
    base_layer: nn.Linear,
    rank: int = 8,
    scaling: float = 2.0,
    kind: int = HEAD,
    out_group_size: int = 1,
    mask_in: Optional[torch.Tensor] = None,
    mask_out: Optional[torch.Tensor] = None,
    share_input_mask: Optional[torch.Tensor] = None,
    layer_idx: int = -1,
    module_name: str = "",
    cache_for_salience: bool = False,
    w_a_std: Optional[float] = None,
) -> MaskedLinear:
    """Build a :class:`MaskedLinear` around ``base_layer`` with a fresh adapter.

    ``share_input_mask`` lets several modules share the *same* ``m_i`` tensor object
    (hidden-dimension pruning of the residual stream); ``mask_in`` is used otherwise.
    """
    in_mask = share_input_mask if share_input_mask is not None else mask_in
    if in_mask is not None:
        in_mask = in_mask.detach().clone().to(torch.float32).reshape(-1)
    adapter = APTAdapter(
        in_features=base_layer.in_features,
        out_features=base_layer.out_features,
        rank=rank,
        scaling=scaling,
        w_a_std=w_a_std,
        dtype=base_layer.weight.dtype,
        device=base_layer.weight.device,
    )
    return MaskedLinear(
        base_layer=base_layer,
        adapter=adapter,
        mask_in=in_mask,
        mask_out=mask_out,
        kind=kind,
        out_group_size=out_group_size,
        layer_idx=layer_idx,
        module_name=module_name,
        cache_for_salience=cache_for_salience,
    )


def iter_masked_linears(module: nn.Module, names: Optional[Sequence[str]] = None):
    """Yield ``(name, MaskedLinear)`` pairs below ``module`` (optional name filter)."""
    for name, child in module.named_modules():
        if isinstance(child, MaskedLinear):
            if names is None or any(n in name for n in names):
                yield name, child


def masked_linears_by_layer(module: nn.Module) -> dict:
    """Group :class:`MaskedLinear` modules by their transformer ``layer_idx``."""
    out: dict = {}
    for name, child in module.named_modules():
        if isinstance(child, MaskedLinear):
            out.setdefault(child.layer_idx, []).append((name, child))
    return out


def reset_zero_ranks(module: nn.Module) -> None:
    """Ensure every adapter has at least rank 1 (safety helper for schedules)."""
    for child in module.modules():
        if isinstance(child, MaskedLinear) and child.adapter is not None:
            if child.adapter.rank <= 0:
                child.adapter.increase_rank(1)
