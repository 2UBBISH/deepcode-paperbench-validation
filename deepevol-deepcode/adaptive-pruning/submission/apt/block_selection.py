"""Latency-saliency knapsack for APT block selection.

Implements the "Efficient search of LM block parameters" procedure of the paper
(Section 4.2) together with its full details in Appendix C:

* the approximated LM parameter count of Eq. (6)::

      C(Theta_t ; M_t) ~= d_m * sum_i ( 4 * n_h^i * d_h + 2 * n_f^i )

* the parameter count of a single block (Appendix C, first display equation)::

      C_head      = 4 * d_m * d_m / n_h
      C_neuron    = 2 * d_m
      C_dimension = n_L * (4 * d_m + 2 * n_f)

* the block-type function ``f(b)`` (Appendix C, second display equation)::

      f(b_i) = 0 if head, 1 if neuron, 2 if dimension

* the parameter count of the model consisting of the top-i blocks (third
  display equation)::

      C_top-i = (4 * d_h' * n_h' + 2 * n_f') * d_m'
      n_h' = sum_{j<i} delta(0, f(b_j))
      n_f' = sum_{j<i} delta(1, f(b_j))
      d_m' = sum_{j<i} delta(2, f(b_j))

* blocks are sorted by *salience density* (salience / number of parameters of
  the block) and the largest feasible top-i is found with a binary search,
  because C_top-i is monotonically non-decreasing in i as long as every block
  contributes a non-negative parameter count ("the parameter size monotonically
  increases with block quantity").

Paper details reproduced here:

* "We also omit the bias term for density calculation since it takes up less
  than 1% of LM's parameters."
* "Since the number of heads, neurons, and hidden dimensions is ever-changing
  during pruning, we re-calculate the density after executing each parameter
  size change."  -> :func:`BlockSelector.recompute_densities`
* "for T5 and LLaMA-like models, the FFN layers are gated, consisting of up-,
  gate-, and down-projection linear layers. Therefore, the number of layers in
  FFN shall be three instead of two in these LMs."  -> ``ffn_linear_count=3``
* "for encoder-decoder LMs like T5, the cross-attention layers in the decoder
  shall also be counted."  -> ``n_cross_attn_layers``
* "We leave the implementation details in Appendix C" -> binary-search knapsack
  below.

Not implemented here (deliberately): the mask *update* itself.  Appendix C
states that pruned blocks' masks are "gradually decreased by alpha = 0.01"
rather than set to 0; the gradual decay lives in :mod:`apt.masks` and consumes
the ``SelectionResult`` produced by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # the block-type constants are shared with the adapters (Appendix C)
    from .adapters import BLOCK_TYPES, DIMENSION, HEAD, NEURON
except Exception:  # pragma: no cover - fallback keeps this module importable
    HEAD, NEURON, DIMENSION = 0, 1, 2
    BLOCK_TYPES = {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}

__all__ = [
    "HEAD",
    "NEURON",
    "DIMENSION",
    "BLOCK_TYPES",
    "ModelShape",
    "Block",
    "SelectionResult",
    "BlockSelector",
    "ffn_linear_count_for",
    "head_param_count",
    "neuron_param_count",
    "dimension_param_count",
    "approx_param_count",
    "sort_by_density",
    "binary_search_top_i",
    "select_blocks",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_float(value) -> float:
    """Best-effort conversion of python/numpy/torch scalars to ``float``."""
    if value is None:
        return 0.0
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    for attr in ("item", "detach", "cpu", "numpy"):
        if hasattr(value, attr):
            try:
                return _to_float(getattr(value, attr)())
            except Exception:  # pragma: no cover
                break
    try:
        return float(value)
    except Exception:  # pragma: no cover
        return 0.0


def ffn_linear_count_for(model_type: str) -> int:
    """Appendix C: gated FFNs (T5 / LLaMA-like) count as three linear layers.

    Non gated FFNs (BERT / RoBERTa) count as two (up- and down-projection).
    """
    model_type = (model_type or "").lower()
    if model_type in {"t5", "mt5", "llama", "llama2", "llama-2", "opt", "gpt2"}:
        return 3
    return 2


# --------------------------------------------------------------------------- #
# parameter-count estimates (Appendix C, Eq. (6))
# --------------------------------------------------------------------------- #
def head_param_count(d_model: float, n_heads: int, attn_linear_count: int = 4,
                     head_dim: Optional[float] = None) -> float:
    """``C_head = 4 * d_m * d_m / n_h`` (paper example: 4*768*768/12 = 196608).

    ``head_dim`` may be supplied explicitly; otherwise the per-head dimension is
    ``d_m / n_h`` exactly as in the paper's illustration.
    """
    if n_heads <= 0:
        return 0.0
    d_h = d_model / float(n_heads) if head_dim is None else float(head_dim)
    return float(attn_linear_count) * float(d_model) * d_h


def neuron_param_count(d_model: float, ffn_linear_count: int = 2) -> float:
    """``C_neuron = 2 * d_m`` (3 * d_m for gated FFNs)."""
    return float(ffn_linear_count) * float(d_model)


def dimension_param_count(d_model: float, n_ffn: float, n_layers: int,
                          ffn_linear_count: int = 2, attn_linear_count: int = 4,
                          n_cross_attn_layers: int = 0) -> float:
    """``C_dimension = n_L * (4 * d_m + 2 * n_f)`` per hidden dimension.

    Every attention site contributes ``attn_linear_count * d_m`` (removing a
    hidden dimension drops one column/row of the q/k/v/o projections) and every
    FFN adds ``ffn_linear_count * n_f``.  Encoder-decoder models add the
    decoder's cross-attention sites through ``n_cross_attn_layers``.
    """
    n_attn_sites = float(n_layers) + float(n_cross_attn_layers)
    n_ffn_sites = float(n_layers)
    return float(attn_linear_count) * float(d_model) * n_attn_sites + \
        float(ffn_linear_count) * float(n_ffn) * n_ffn_sites


def approx_param_count(n_heads: float, n_ffn: float, d_model: float,
                       head_dim: float, ffn_linear_count: int = 2,
                       attn_linear_count: int = 4) -> float:
    """Eq. (6) / ``C_top-i`` with aggregated (already summed) block counts.

    ``n_heads`` and ``n_ffn`` are *totals over the whole LM* (i.e. already summed
    over layers), while ``d_model`` is the number of retained hidden dimensions.
    Bias terms and layer norms are ignored, and tuning parameters are not counted
    because they can be fully merged after training (footnote 2 of Section 4.2).
    """
    if d_model <= 0:
        return 0.0
    return (float(attn_linear_count) * float(head_dim) * float(n_heads)
            + float(ffn_linear_count) * float(n_ffn)) * float(d_model)


# --------------------------------------------------------------------------- #
# model shape description
# --------------------------------------------------------------------------- #
@dataclass
class ModelShape:
    """Static (pre-pruning) description of the LM that is being pruned.

    ``d_model``, ``n_heads`` and ``n_ffn`` are the *original* sizes; the current
    (retained) sizes are tracked in :class:`Counts` inside :class:`BlockSelector`
    because "the number of heads, neurons, and hidden dimensions is
    ever-changing during pruning".
    """

    d_model: int
    n_layers: int
    n_heads: int
    n_ffn: int
    ffn_linear_count: int = 2
    attn_linear_count: int = 4
    n_cross_attn_layers: int = 0
    scale_head_dim_with_d_model: bool = True
    include_cross_attn_heads: bool = True
    model_type: str = ""

    # -- derived quantities ------------------------------------------------- #
    @property
    def head_dim(self) -> float:
        """Per-head dimension ``d_h = d_m / n_h`` (RoBERTa-base: 64)."""
        if self.n_heads <= 0:
            return 0.0
        return self.d_model / float(self.n_heads)

    @property
    def n_attn_sites(self) -> int:
        cross = self.n_cross_attn_layers if self.include_cross_attn_heads else 0
        return int(self.n_layers) + int(cross)

    @property
    def total_heads(self) -> int:
        """Total number of MHA head blocks in the LM."""
        return int(self.n_attn_sites) * int(self.n_heads)

    @property
    def total_neurons(self) -> int:
        """Total number of FFN neuron blocks in the LM."""
        return int(self.n_layers) * int(self.n_ffn)

    @property
    def total_dims(self) -> int:
        return int(self.d_model)

    def head_dim_at(self, d_model: float) -> float:
        """Head dimension after the hidden size shrank to ``d_model``."""
        if self.scale_head_dim_with_d_model and self.n_heads > 0:
            return float(d_model) / float(self.n_heads)
        return self.head_dim

    def head_block_cost(self, d_model: float) -> float:
        return float(self.attn_linear_count) * float(d_model) * \
            self.head_dim_at(d_model)

    def neuron_block_cost(self, d_model: float) -> float:
        return neuron_param_count(d_model, self.ffn_linear_count)

    def dim_block_cost(self, d_model: float, n_ffn: float) -> float:
        return dimension_param_count(
            d_model, n_ffn, self.n_layers,
            ffn_linear_count=self.ffn_linear_count,
            attn_linear_count=self.attn_linear_count,
            n_cross_attn_layers=self.n_cross_attn_layers,
        )

    def reference_block_costs(self) -> Dict[str, float]:
        """The three reference numbers of Appendix C (RoBERTa-base example)."""
        return {
            "head": self.head_block_cost(self.d_model),
            "neuron": self.neuron_block_cost(self.d_model),
            "dimension": self.dim_block_cost(self.d_model, self.n_ffn),
        }

    def full_param_count(self) -> float:
        """Eq. (6) with everything retained (RoBERTa-base ~= 84.9M)."""
        return approx_param_count(
            n_heads=self.total_heads,
            n_ffn=self.total_neurons,
            d_model=self.total_dims,
            head_dim=self.head_dim,
            ffn_linear_count=self.ffn_linear_count,
            attn_linear_count=self.attn_linear_count,
        )


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """One prunable parameter block: an MHA head, an FFN neuron or a hidden dim.

    ``kind`` is exactly the paper's ``f(b_i)`` (0 head, 1 neuron, 2 dimension).
    ``site`` distinguishes self- from cross-attention heads for encoder-decoder
    LMs (e.g. ``"self"`` / ``"cross"``).
    """

    kind: int
    layer: int
    index: int
    site: str = ""
    salience: float = 0.0
    param_count: float = 0.0

    @property
    def density(self) -> float:
        """Salience density: block salience / number of parameters in the block."""
        if self.param_count <= 0:
            return float("inf") if self.salience > 0 else 0.0
        return self.salience / self.param_count

    @property
    def kind_name(self) -> str:
        return BLOCK_TYPES.get(self.kind, str(self.kind))

    @property
    def name(self) -> str:
        base = f"{self.kind_name}.{self.layer}.{self.index}"
        return f"{base}.{self.site}" if self.site else base

    def sort_key(self) -> Tuple:
        """Deterministic tie-break used after the density sort."""
        return (self.kind, self.layer, self.site, self.index)


@dataclass
class SelectionResult:
    """Outcome of the latency-saliency knapsack search."""

    retained: List[Block] = field(default_factory=list)
    pruned: List[Block] = field(default_factory=list)
    order: List[Block] = field(default_factory=list)
    n_heads: int = 0
    n_neurons: int = 0
    d_model: int = 0
    param_count: float = 0.0
    target_param_count: float = 0.0
    original_param_count: float = 0.0
    n_top: int = 0

    @property
    def sparsity(self) -> float:
        if self.original_param_count <= 0:
            return 0.0
        return 1.0 - self.param_count / self.original_param_count

    def mask_for(self, block: Block) -> float:
        """1.0 if the block is retained (retained mask), else 0.0."""
        return 1.0 if block in self.retained else 0.0

    def retained_names(self) -> List[str]:
        return [b.name for b in self.retained]

    def pruned_names(self) -> List[str]:
        return [b.name for b in self.pruned]

    def as_masks(self, shape: ModelShape) -> Dict[str, object]:
        """Group the binary decision per (kind, layer) for :mod:`apt.masks`.

        Returns ``{"heads": {layer: [0/1 per head]}, "neurons": {...},
        "dims": [0/1 per hidden dim]}``; cross-attention heads are keyed by
        ``(layer, "cross")``.
        """
        keep = {b.name for b in self.retained}
        heads: Dict[object, List[float]] = {}
        neurons: Dict[int, List[float]] = {}
        dims = [0.0] * int(shape.d_model)

        for layer in range(int(shape.n_layers)):
            sites: Sequence[str] = ("self",) if shape.n_cross_attn_layers == 0 \
                else ("self",) + tuple(["cross"] * int(shape.n_cross_attn_layers > 0))
            for site in sites:
                if site == "cross" and (not shape.include_cross_attn_heads or layer >= shape.n_cross_attn_layers):
                    continue
                key = layer if site == "self" else (layer, site)
                heads[key] = [
                    1.0 if Block(HEAD, layer, h, site).name in keep else 0.0
                    for h in range(int(shape.n_heads))
                ]
            neurons[layer] = [
                1.0 if Block(NEURON, layer, j).name in keep else 0.0
                for j in range(int(shape.n_ffn))
            ]
        for d in range(int(shape.d_model)):
            dims[d] = 1.0 if Block(DIMENSION, -1, d).name in keep else 0.0
        return {"heads": heads, "neurons": neurons, "dims": dims}


# --------------------------------------------------------------------------- #
# sorting + binary search
# --------------------------------------------------------------------------- #
def sort_by_density(blocks: Iterable[Block]) -> List[Block]:
    """Sort blocks by salience density (descending); deterministic tie-break.

    "we first sort the blocks by their salience divided by the parameter number"
    """
    return sorted(blocks, key=lambda b: (-b.density,) + b.sort_key())


def binary_search_top_i(n_heads_prefix: Sequence[int],
                        n_ffn_prefix: Sequence[int],
                        n_dim_prefix: Sequence[int],
                        shape: ModelShape,
                        target_param_count: float,
                        n_blocks: Optional[int] = None,
                        feasible=None) -> int:
    """Largest ``i`` whose top-i blocks fit into ``target_param_count``.

    Uses the closed form ``C_top-i = (4 d_h' n_h' + 2 n_f') * d_m'`` evaluated
    from prefix counts, so each probe of the binary search is O(1).

    Parameters
    ----------
    n_heads_prefix / n_ffn_prefix / n_dim_prefix:
        Prefix sums of the Kronecker deltas ``delta(0, f(b_j))``,
        ``delta(1, f(b_j))`` and ``delta(2, f(b_j))`` over the density-sorted
        block list (length N+1).
    target_param_count:
        Parameter budget the retained top-i blocks must not exceed.
    feasible:
        Optional predicate ``feasible(i, n_heads, n_ffn, n_dim) -> bool`` used
        for additional (monotone) structural constraints.

    Returns
    -------
    int
        Number of blocks (``i``) to retain, i.e. ``order[:i]`` are kept.
    """
    n_blocks = len(n_heads_prefix) - 1 if n_blocks is None else int(n_blocks)

    def ok(i: int) -> bool:
        n_h = n_heads_prefix[i]
        n_f = n_ffn_prefix[i]
        d_m = n_dim_prefix[i]
        c = approx_param_count(
            n_heads=n_h,
            n_ffn=n_f,
            d_model=d_m,
            head_dim=shape.head_dim_at(d_m),
            ffn_linear_count=shape.ffn_linear_count,
            attn_linear_count=shape.attn_linear_count,
        )
        if c > target_param_count:
            return False
        if feasible is not None and not feasible(i, n_h, n_f, d_m):
            return False
        return True

    # C_top-i is monotone non-decreasing in i -> binary search the boundary.
    lo, hi = 0, n_blocks  # ok(0) is always True (C_top-0 = 0)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


# --------------------------------------------------------------------------- #
# selector
# --------------------------------------------------------------------------- #
class BlockSelector:
    """Density sort + binary-search knapsack over head/neuron/dimension blocks.

    Density is re-computed at every call through :meth:`recompute_densities`
    because the number of heads, neurons and hidden dimensions - and therefore
    each block's parameter count - changes during pruning (Appendix C).
    """

    def __init__(self, shape: ModelShape, ffn_linear_count: Optional[int] = None,
                 attn_linear_count: Optional[int] = None):
        self.shape = shape
        if ffn_linear_count is not None:
            self.shape.ffn_linear_count = int(ffn_linear_count)
        if attn_linear_count is not None:
            self.shape.attn_linear_count = int(attn_linear_count)

    # -- block enumeration -------------------------------------------------- #
    def enumerate_blocks(self, salience: Optional[Dict[str, float]] = None) -> List[Block]:
        """Build the full block list of the LM described by ``self.shape``.

        ``salience`` optionally maps a block name (see :attr:`Block.name`) to its
        outlier-aware salience score computed by :mod:`apt.salience`.
        """
        salience = salience or {}
        blocks: List[Block] = []
        shape = self.shape
        for layer in range(int(shape.n_layers)):
            for h in range(int(shape.n_heads)):
                for site in self._head_sites(layer):
                    b = Block(HEAD, layer, h, site=site)
                    blocks.append(self._with_salience(b, salience))
            for j in range(int(shape.n_ffn)):
                b = Block(NEURON, layer, j)
                blocks.append(self._with_salience(b, salience))
        for d in range(int(shape.d_model)):
            b = Block(DIMENSION, -1, d)
            blocks.append(self._with_salience(b, salience))
        return blocks

    def _head_sites(self, layer: int) -> Tuple[str, ...]:
        shape = self.shape
        if shape.n_cross_attn_layers > 0 and shape.include_cross_attn_heads \
                and layer < int(shape.n_cross_attn_layers):
            return ("self", "cross")
        return ("self",)

    @staticmethod
    def _with_salience(block: Block, salience: Dict[str, float]) -> Block:
        if block.name in salience:
            block.salience = _to_float(salience[block.name])
        return block

    # -- density bookkeeping ------------------------------------------------ #
    def recompute_densities(self, blocks: Sequence[Block],
                            counts: Optional[Dict[str, float]] = None) -> List[Block]:
        """Refresh each block's parameter count/density after a size change.

        "Since the number of heads, neurons, and hidden dimensions is
        ever-changing during pruning, we re-calculate the density after executing
        each parameter size change."  Bias terms are omitted (Appendix C).
        """
        counts = counts or {}
        # If a subset is pruned, pruned blocks are no longer part of the LM, so
        # the remaining blocks become *relatively* more important; the per-block
        # cost itself depends on the currently retained hidden size / neurons.
        d_m = float(counts.get("d_model", self.shape.d_model))
        n_ffn = float(counts.get("n_ffn", self.shape.n_ffn))
        for b in blocks:
            if b.kind == HEAD:
                b.param_count = self.shape.head_block_cost(d_m)
            elif b.kind == NEURON:
                b.param_count = self.shape.neuron_block_cost(d_m)
            elif b.kind == DIMENSION:
                b.param_count = self.shape.dim_block_cost(d_m, n_ffn)
        return list(blocks)

    # -- the knapsack itself ------------------------------------------------ #
    def select(self, blocks: Sequence[Block],
               target_param_count: float,
               original_param_count: Optional[float] = None,
               feasible=None) -> SelectionResult:
        """Retain the top-i salient blocks such that C_top-i <= budget.

        The remaining blocks are the ones to be pruned; Appendix C notes that
        their masks are annealed by alpha = 0.01 instead of being zeroed, which
        is handled by :mod:`apt.masks`.
        """
        order = sort_by_density(blocks)

        n_heads_prefix = [0]
        n_ffn_prefix = [0]
        n_dim_prefix = [0]
        for b in order:
            n_heads_prefix.append(n_heads_prefix[-1] + (1 if b.kind == HEAD else 0))
            n_ffn_prefix.append(n_ffn_prefix[-1] + (1 if b.kind == NEURON else 0))
            n_dim_prefix.append(n_dim_prefix[-1] + (1 if b.kind == DIMENSION else 0))

        n_top = binary_search_top_i(
            n_heads_prefix, n_ffn_prefix, n_dim_prefix,
            shape=self.shape,
            target_param_count=target_param_count,
            feasible=feasible,
        )
        retained = order[:n_top]
        pruned = order[n_top:]
        param_count = approx_param_count(
            n_heads=n_heads_prefix[n_top],
            n_ffn=n_ffn_prefix[n_top],
            d_model=n_dim_prefix[n_top],
            head_dim=self.shape.head_dim_at(n_dim_prefix[n_top]),
            ffn_linear_count=self.shape.ffn_linear_count,
            attn_linear_count=self.shape.attn_linear_count,
        )
        return SelectionResult(
            retained=retained,
            pruned=pruned,
            order=order,
            n_heads=n_heads_prefix[n_top],
            n_neurons=n_ffn_prefix[n_top],
            d_model=n_dim_prefix[n_top],
            param_count=param_count,
            target_param_count=target_param_count,
            original_param_count=original_param_count
            if original_param_count is not None else self.shape.full_param_count(),
            n_top=n_top,
        )

    def select_to_sparsity(self, blocks: Sequence[Block], sparsity: float,
                           original_param_count: Optional[float] = None,
                           feasible=None) -> SelectionResult:
        """Same as :meth:`select` but with a *sparsity* target ``gamma_t``.

        The constraint is Eq. (1)'s ``C(Theta_t; M_t) <= (1 - gamma_t) C_0``.
        """
        c0 = self.shape.full_param_count() if original_param_count is None \
            else float(original_param_count)
        target = (1.0 - float(sparsity)) * c0
        return self.select(blocks, target, original_param_count=c0, feasible=feasible)


# --------------------------------------------------------------------------- #
# convenience wrappers
# --------------------------------------------------------------------------- #
def select_blocks(salience: Dict[str, float], shape: ModelShape,
                  sparsity: float,
                  ffn_linear_count: Optional[int] = None,
                  counts: Optional[Dict[str, float]] = None,
                  feasible=None) -> SelectionResult:
    """One-shot helper: enumerate -> refresh density -> knapsack search."""
    selector = BlockSelector(shape, ffn_linear_count=ffn_linear_count)
    blocks = selector.enumerate_blocks(salience)
    selector.recompute_densities(blocks, counts)
    return selector.select_to_sparsity(blocks, sparsity, feasible=feasible)


def roberta_base_shape() -> ModelShape:
    """The RoBERTa-base shape used for the sanity numbers of Appendix C."""
    return ModelShape(
        d_model=768, n_layers=12, n_heads=12, n_ffn=3072,
        ffn_linear_count=2, attn_linear_count=4,
        n_cross_attn_layers=0, model_type="roberta",
    )


def t5_base_shape() -> ModelShape:
    """T5-base-like encoder-decoder shape (gated FFN + decoder cross-attention)."""
    return ModelShape(
        d_model=768, n_layers=24, n_heads=12, n_ffn=3072,
        ffn_linear_count=3, attn_linear_count=4,
        n_cross_attn_layers=12, model_type="t5",
    )


def _selftest() -> None:  # pragma: no cover - exercised via ``python -m``
    shape = roberta_base_shape()
    costs = shape.reference_block_costs()
    assert round(costs["head"]) == 196608, costs
    assert round(costs["neuron"]) == 1536, costs
    assert round(costs["dimension"]) == 110592, costs

    selector = BlockSelector(shape)
    blocks = selector.enumerate_blocks()
    assert len(blocks) == shape.total_heads + shape.total_neurons + shape.total_dims
    selector.recompute_densities(blocks)

    # uniform salience -> density purely driven by block size
    res = selector.select_to_sparsity(blocks, 0.6)
    print("blocks:", len(blocks), "retained:", res.n_top,
          "heads:", res.n_heads, "neurons:", res.n_neurons, "dims:", res.d_model)
    print("C_top-i: %.0f  target: %.0f  sparsity: %.3f  C0: %.0f"
          % (res.param_count, res.target_param_count, res.sparsity,
             res.original_param_count))
    assert res.param_count <= res.target_param_count + 1e-6
    assert res.sparsity >= 0.0
    print("roberta-base C0 = %.2fM" % (res.original_param_count / 1e6))
    assert 80e6 < res.original_param_count < 90e6


if __name__ == "__main__":  # pragma: no cover
    _selftest()
