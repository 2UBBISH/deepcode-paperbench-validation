"""Prunable-block bookkeeping, outlier-aware salience and mask search.

Implements Section 4.2 ("Low-cost Adaptive LM Pruning") together with the
details of Appendix B and Appendix C of the paper.

Three families of structured blocks are pruned simultaneously:

=====  ==============================  =====================================
kind   physical weight slices          paper's parameter count
=====  ==============================  =====================================
head   out-rows of Q/K/V at a head     4 * d_h * d_m       (per layer)
       in-cols of O at a head
neuron out-row of the 1st FFN mat      k_ff * d_m          (k_ff = 2 or 3)
       in-col of the last FFN mat
dim    in-cols of Q/K/V/O and of the   4 * d_m + k_ff * n_f (per layer)
       first FFN mat, out-rows of O
       and of the last FFN mat
=====  ==============================  =====================================

Every feature of every wrapped linear belongs to exactly one block, which lets
us write the per-linear masks from a single vector of block mask values.

Salience
--------
For a block ``b``:

    S(b) = sum over slices of [ compressed frozen salience ] + [ tuning salience ]

with (Appendix B)

    compressed frozen salience of an input dim  j = (sum |dL/dX_j|) * (sum |X_j|)
    compressed frozen salience of an output dim i = (sum |dL/dH_i|) * (sum |H_i|)
    tuning salience                              = |W_A . dL/dW_A| / |W_B . dL/dW_B|

and the *outlier-aware* score (Eq. 5)

    S_hat(b) = S(b) + sqrt(Kurt(activation of the block's features))

where the kurtosis is computed from streaming moments of the activation
distribution over the batch x sequence axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# --------------------------------------------------------------------------- #
# Block description
# --------------------------------------------------------------------------- #
HEAD, NEURON, DIM = 0, 1, 2


@dataclass
class ParamSlice:
    """A contiguous run of feature indices of one wrapped linear layer."""

    linear: str          # key into ``Topology.linears``
    dim: str             # "in" or "out"
    start: int
    size: int            # >1 only for attention heads (a run of d_h channels)

    @property
    def stop(self) -> int:
        return self.start + self.size


@dataclass
class Block:
    """One prunable structural unit (an MHA head, an FFN neuron or a dim)."""

    bid: str
    kind: int
    group: str                       # e.g. "enc" / "dec" for encoder-decoder LMs
    slices: List[ParamSlice] = field(default_factory=list)
    param_count: int = 0

    def add(self, linear: str, dim: str, start: int, size: int = 1) -> None:
        self.slices.append(ParamSlice(linear, dim, start, size))


# --------------------------------------------------------------------------- #
# Salience bookkeeping
# --------------------------------------------------------------------------- #
class PruningState:
    """Holds the block table, the running salience and the pruning masks."""

    def __init__(
        self,
        blocks: Sequence[Block],
        d_h: int,
        ema_beta: float = 0.85,
        alpha: float = 0.01,
        count_mode: str = "formula",
    ) -> None:
        self.blocks: List[Block] = list(blocks)
        self.n = len(self.blocks)
        self.d_h = int(d_h)
        self.ema_beta = float(ema_beta)
        self.alpha = float(alpha)
        self.count_mode = count_mode

        self.salience_ema = torch.zeros(self.n, dtype=torch.float64)
        self._ema_initialised = False
        # ``mask`` is a *soft* mask in [0, 1] that is annealed by ``alpha``
        # towards the hard 0/1 decision (Appendix C).  It starts from a fully
        # dense model so that the first training steps see the unpruned LM.
        self.mask = torch.ones(self.n, dtype=torch.float64)
        self.retain = torch.ones(self.n, dtype=torch.bool)

        counts = torch.tensor([b.param_count for b in self.blocks], dtype=torch.float64)
        self._block_costs = counts
        self._kinds = torch.tensor([b.kind for b in self.blocks], dtype=torch.long)
        #: groups that own hidden-dimension blocks (one for RoBERTa/BERT, the
        #: single shared dimension of T5)
        self._dim_groups = sorted({b.group for b in self.blocks if b.kind == DIM})
        self._ffn_mult = 3 if any(
            b.kind == NEURON and len(b.slices) == 3 for b in self.blocks
        ) else 2

    # ------------------------------------------------------------------ info
    @property
    def total_parameters(self) -> float:
        """Number of parameters of the *retained* model at full density.

        Note that head/neuron blocks partition the *rows* of a weight matrix
        while dimension blocks partition its *columns*; summing block sizes
        would therefore double count.  The paper's Eq. (6) cardinality formula
        is used instead, which agrees with the true model size at full density
        and gives the size of the retained sub-network otherwise.
        """
        n_head = sum(1 for b in self.blocks if b.kind == HEAD)
        n_neuron = sum(1 for b in self.blocks if b.kind == NEURON)
        n_dim = sum(1 for b in self.blocks if b.kind == DIM)
        if n_dim == 0:
            return 0.0
        return float(n_dim * (4.0 * self.d_h * n_head + self._ffn_mult * n_neuron))

    def current_sparsity(self, hard: bool = True) -> float:
        """Fraction of the countable LM parameters that is currently pruned."""
        retained = self.retain if hard else (self.mask >= 0.5)
        n_head = int((retained & (self._kinds == HEAD)).sum().item())
        n_neuron = int((retained & (self._kinds == NEURON)).sum().item())
        n_dim = int((retained & (self._kinds == DIM)).sum().item())
        kept = n_dim * (4.0 * self.d_h * n_head + self._ffn_mult * n_neuron)
        total = self.total_parameters
        if total <= 0:
            return 0.0
        return 1.0 - kept / total

    # ------------------------------------------------------------ parameter #
    def parameters_for_prefix(self, k: int, order: torch.Tensor) -> float:
        """Parameter count of the model that keeps the ``top-k`` scored blocks.

        Implements the paper's Eq. (6)::

            C(top-i) = sum_g d_g' * (4 * d_h * n_h^g + k_ff * n_f^g)

        where ``d_g'``, ``n_h^g`` and ``n_f^g`` are the number of retained
        dimension, head and neuron blocks of group ``g``.  The count is exactly
        the size of the retained sub-network (it correctly accounts for the
        fact that heads/neurons partition rows while dimensions partition
        columns) and is strictly increasing in ``k``, which is the monotonicity
        the binary search relies on.
        """
        if k <= 0:
            return 0.0
        return float(self._prefix_costs(order)[k])

    # --------------------------------------------------------- cost prefixes #
    def _prefix_costs(self, order: torch.Tensor) -> torch.Tensor:
        """``[C(0), C(1), ..., C(n)]`` for the given block ordering.

        ``C(k)`` is the size of the sub-network that keeps the ``top-k`` scored
        blocks.  It is built from the increments

            Delta C(head)   = 4 d_h * d_g
            Delta C(neuron) = k_ff  * d_g
            Delta C(dim)    = 4 d_h * n_h^g + k_ff * n_f^g

        where the counts are the ones *before* the block is added.  Everything is
        computed with vectorised cumulative sums, so the routine is cheap enough
        to be called at every training step (Algorithm 1 re-selects the blocks
        each step).
        """
        cached = getattr(self, "_prefix_cache", None)
        key = hash(order.numpy().tobytes())
        if cached is not None and cached[0] == key and cached[1].numel() == self.n + 1:
            return cached[1]

        import numpy as np

        kinds = self._kinds.numpy()
        if not hasattr(self, "_group_ids"):
            names = sorted({b.group for b in self.blocks})
            self._group_names = names
            self._group_ids = {g: i for i, g in enumerate(names)}
        groups = np.array([self._group_ids[b.group] for b in self.blocks], dtype=np.int64)

        order_np = order.numpy().astype(np.int64)
        k_sorted = kinds[order_np]
        g_sorted = groups[order_np]

        delta = np.zeros(self.n, dtype=np.float64)
        for gi in range(len(self._group_names)):
            in_group = g_sorted == gi
            is_head = (k_sorted == HEAD) & in_group
            is_neuron = (k_sorted == NEURON) & in_group
            is_dim = (k_sorted == DIM) & in_group
            counts_before = lambda mask: np.concatenate(([0], np.cumsum(mask)[:-1]))  # noqa: E731
            d_before = counts_before(is_dim)
            h_before = counts_before(is_head)
            n_before = counts_before(is_neuron)
            delta += is_head * (4.0 * self.d_h * d_before)
            delta += is_neuron * (self._ffn_mult * d_before)
            delta += is_dim * (4.0 * self.d_h * h_before + self._ffn_mult * n_before)

        costs = torch.zeros(self.n + 1, dtype=torch.float64)
        costs[1:] = torch.from_numpy(np.cumsum(delta))
        self._prefix_cache = (key, costs)
        return costs

    # -------------------------------------------------------- salience score
    def update_salience(self, scores: torch.Tensor) -> None:
        """Exponential moving average of the per-block outlier-aware salience."""
        scores = scores.detach().to(torch.float64).cpu()
        if not self._ema_initialised:
            self.salience_ema = scores.clone()
            self._ema_initialised = True
        else:
            b = self.ema_beta
            self.salience_ema = b * self.salience_ema + (1.0 - b) * scores

    def sort_order(self) -> torch.Tensor:
        """Blocks sorted by *salience density* (salience / parameter count)."""
        density = self.salience_ema / self._block_costs.clamp_min(1.0)
        return torch.argsort(density, descending=True)

    # ------------------------------------------------------------ mask search
    def select_for_budget(self, keep_ratio: float) -> torch.Tensor:
        """Binary search for the top salient blocks under a parameter budget.

        ``keep_ratio`` is the fraction of the *original* LM parameters that may
        be retained; the returned boolean vector marks the retained blocks.

        This is the latency-saliency knapsack of the paper reduced to a
        monotone prefix search (Appendix C).
        """
        order = self.sort_order()
        budget = max(0.0, float(keep_ratio)) * self.total_parameters

        costs = self._prefix_costs(order)
        # ``costs`` is non-decreasing, so a binary search over k is exact.
        lo, hi = 0, self.n
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if float(costs[mid].item()) <= budget + 1e-9:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        retain = torch.zeros(self.n, dtype=torch.bool)
        if best > 0:
            retain[order[:best]] = True
        self.retain = retain
        return retain

    def anneal_masks(self) -> None:
        """Gradually move the soft masks towards the hard 0/1 decision.

        Appendix C: "we do not set the pruned blocks' corresponding masks to 0
        directly but gradually decrease their values by ``alpha = 0.01``".
        """
        target = self.retain.to(torch.float64)
        up = torch.minimum(torch.ones_like(self.mask), self.mask + self.alpha)
        down = torch.maximum(torch.zeros_like(self.mask), self.mask - self.alpha)
        self.mask = torch.where(target > 0.5, up, down)

    def harden_masks(self) -> None:
        self.mask = self.retain.to(torch.float64)

    # ------------------------------------------------------------- writeback
    def write_masks_to_linears(self, linears: Dict[str, torch.nn.Module]) -> None:
        """Broadcast the per-block mask values onto every wrapped linear."""
        for name, lin in linears.items():
            lin.mask_in.fill_(1.0)
            lin.mask_out.fill_(1.0)
        for i, block in enumerate(self.blocks):
            v = float(self.mask[i].item())
            for sl in block.slices:
                lin = linears[sl.linear]
                buf = lin.mask_in if sl.dim == "in" else lin.mask_out
                buf[sl.start : sl.stop] = v

    # -------------------------------------------------------------- coverage
    def check_coverage(self, linears: Dict[str, torch.nn.Module]) -> None:
        """Sanity check: every feature of every wrapped linear is covered once."""
        counts: Dict[Tuple[str, str], torch.Tensor] = {}
        for name, lin in linears.items():
            counts[(name, "in")] = torch.zeros(lin.in_features, dtype=torch.long)
            counts[(name, "out")] = torch.zeros(lin.out_features, dtype=torch.long)
        for block in self.blocks:
            for sl in block.slices:
                counts[(sl.linear, sl.dim)][sl.start : sl.stop] += 1
        for key, c in counts.items():
            if not bool((c == 1).all()):
                bad = int((c != 1).sum().item())
                raise RuntimeError(
                    f"block table does not partition {key}: {bad} uncovered/overlapping features"
                )


# --------------------------------------------------------------------------- #
# Kurtosis utilities
# --------------------------------------------------------------------------- #
def kurtosis_from_moments(moments: torch.Tensor, count: float) -> torch.Tensor:
    """Pearson kurtosis from streaming raw moments ``[m1, m2, m3, m4]``.

    ``moments`` has shape ``[4, d]`` and stores ``sum x``, ``sum x^2``,
    ``sum x^3`` and ``sum x^4`` accumulated in float64.  The central fourth
    moment is recovered with the binomial expansion::

        mu4 = m4 - 4 m1 m3 + 6 m1^2 m2 - 3 m1^4
        var = m2 - m1^2
        kurt = mu4 / var^2

    Kurtosis is scale invariant, so the (large) activation magnitudes do not
    affect the ranking as long as float64 accumulation is used.
    """
    if count <= 0:
        return torch.zeros_like(moments[0])
    n = float(count)
    m1 = moments[0] / n
    m2 = moments[1] / n
    m3 = moments[2] / n
    m4 = moments[3] / n
    var = (m2 - m1 * m1).clamp_min(0.0)
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.pow(4)
    return mu4 / var.pow(2).clamp_min(1e-30)


def parse_block_kind(name: str) -> int:
    if name.startswith("head"):
        return HEAD
    if name.startswith("neuron"):
        return NEURON
    return DIM


def params_of_block(kind: int, d_m: int, d_h: int, n_f: int, k_ff: int = 2) -> int:
    """Analytic parameter count of a single block (used for tests/reporting)."""
    if kind == HEAD:
        return 4 * d_h * d_m
    if kind == NEURON:
        return k_ff * d_m
    return 4 * d_m + k_ff * n_f


def roberta_base_block_costs(d_m: int = 768, n_h: int = 12, n_f: int = 3072, n_L: int = 12):
    """Reproduce the numbers quoted in Appendix C of the paper."""
    d_h = d_m // n_h
    return {
        "head": 4 * d_m * d_m // n_h,
        "neuron": 2 * d_m,
        "dimension": n_L * (4 * d_m + 2 * n_f),
        "d_h": d_h,
    }


__all__ = [
    "Block",
    "ParamSlice",
    "PruningState",
    "HEAD",
    "NEURON",
    "DIM",
    "kurtosis_from_moments",
    "params_of_block",
    "roberta_base_block_costs",
]
