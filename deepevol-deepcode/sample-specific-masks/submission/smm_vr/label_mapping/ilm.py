"""Iterative Label Mapping (Ilm) for Sample-specific Multi-channel Masks (SMM).

Implements :math:`f_{\\mathrm{out}}^{\\mathrm{Ilm}}` of the SMM paper (ICML 2024).

Paper references
----------------
* §2.3: "Chen et al. (2023) propose iterative label mapping (Ilm) that updates
  :math:`f_{\\mathrm{out}}` in each training iteration, reflecting changes in label
  mapping throughout the learning of :math:`f_{\\mathrm{in}}`."
* Appendix A.4, Eq. (11):

  .. math::

      y_{\\mathrm{Ilm}}^{\\mathrm{P},(j+1)} =
      \\operatorname*{arg\\,max}_{y \\in \\mathcal{Y}^{\\mathrm{P}}}
      \\operatorname*{Pr}\\left\\{ y = f_{\\mathrm{P}}\\left(
      f_{\\mathrm{in}}^{(j)}\\left(x_i \\mid \\theta^{(j)}\\right)\\right)
      \\mid y_i = y^{\\mathrm{T}} \\right\\}

* Appendix A.4 (Algorithm 4): "before training the reprogramming pattern
  :math:`\\theta` in each epoch, Ilm updates the one-to-one mapping from
  :math:`\\mathcal{Y}^{\\mathrm{P}}` to :math:`\\mathcal{Y}^{\\mathrm{T}}` with the
  training samples incorporating the current pattern, iteratively until convergence."
* Appendix A.4 (Algorithm 3): the greedy "find the maximum :math:`d_{y^{\\mathrm{P}},
  y^{\\mathrm{T}}}`" selection with row/column zeroing that guarantees injectivity.

Ilm is therefore exactly the Flm greedy assignment (Algorithm 3) re-run on a
*freshly recomputed* frequency matrix (Algorithm 2, with the current
:math:`f_{\\mathrm{in}}(\\cdot \\mid \\theta^{(j)})`) at the beginning of every epoch.
Like Rlm and Flm it is non-parametric: zero trainable parameters, only an integer
injective mapping stored as module buffers.

This module is the default output mapping for the main tables (Tables 1 and 2) of
the paper.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .flm import (
    IGNORE_INDEX,
    apply_label_mapping,
    flm_mapping_from_frequency_matrix,
)
from .frequency import (
    compute_frequency_matrix,
    frequency_matrix_from_predictions,
)

__all__ = [
    "IlmMapping",
    "IlmLabelMapping",
    "ilm_mapping_from_frequency_matrix",
    "compute_ilm_mapping",
    "ilm_mapping_from_predictions",
    "update_ilm_mapping",
    "build_ilm_mapping",
    "apply_label_mapping",
    "IGNORE_INDEX",
]


# ---------------------------------------------------------------------------
# Mapping construction (Algorithm 2 + Algorithm 3 applied per epoch)
# ---------------------------------------------------------------------------
def ilm_mapping_from_frequency_matrix(
    frequency_matrix: torch.Tensor,
    num_target_classes: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Run the Algorithm 3 greedy selection on a (fresh) frequency matrix.

    Eq. (11) selects, for every target class :math:`y^{\\mathrm{T}}`, the
    pre-trained label :math:`y^{\\mathrm{P}}` that is predicted most often among the
    target training samples of that class.  Algorithm 3 realises this greedily
    (global arg-max of :math:`d`, then zero the selected row *and* column so the
    resulting function stays injective), which is exactly what this function does.

    Because the greedy procedure is identical, this is a thin, well-documented
    wrapper around :func:`smm_vr.label_mapping.flm.flm_mapping_from_frequency_matrix`;
    the *difference* between Flm and Ilm lives in the caller, which decides when to
    recompute ``d`` (once before training for Flm; before every epoch for Ilm).

    Args:
        frequency_matrix: Integer count matrix
            :math:`d \\in \\mathbb{Z}^{|\\mathcal{Y}^{\\mathrm{P}}| \\times |\\mathcal{Y}^{\\mathrm{T}}|}`.
        num_target_classes: Number of target classes; inferred from ``d``'s shape
            when omitted.
        ignore_index: Sentinel used for unmatched pre-trained labels.

    Returns:
        ``dict`` with
        ``target_to_pretrained`` :math:`(|\\mathcal{Y}^{\\mathrm{T}}|,)`,
        ``pretrained_to_target`` :math:`(|\\mathcal{Y}^{\\mathrm{P}}|,)` and the
        post-zeroing ``frequency_matrix``.
    """
    return flm_mapping_from_frequency_matrix(
        frequency_matrix,
        num_target_classes=num_target_classes,
        ignore_index=ignore_index,
    )


def compute_ilm_mapping(
    model: nn.Module,
    data_loader,
    num_target_classes: int,
    num_pretrained_classes: Optional[int] = None,
    f_in=None,
    device: Optional[torch.device] = None,
    *,
    predicted: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
    return_frequency_matrix: bool = False,
) -> Dict[str, torch.Tensor]:
    """One Ilm update: Algorithm 2 with the *current* :math:`f_{\\mathrm{in}}` then Algorithm 3.

    This corresponds to Eq. (11) / Algorithm 4's per-epoch mapping refresh.  The
    frequency matrix must be recomputed with the current reprogramming parameters
    ``f_in`` (i.e. :math:`f_{\\mathrm{in}}^{(j)}(\\cdot \\mid \\theta^{(j)})`), which is
    why ``f_in`` is forwarded to Algorithm 2 on every call.

    Args:
        model: Frozen pre-trained classifier :math:`f_{\\mathrm{P}}`.
        data_loader: Iterable of target *training* batches.
        num_target_classes: :math:`|\\mathcal{Y}^{\\mathrm{T}}|`.
        num_pretrained_classes: :math:`|\\mathcal{Y}^{\\mathrm{P}}|`; inferred from
            ``model`` when omitted.
        f_in: Current input reprogramming function (``None`` means identity,
            matching the zero-initialised pattern :math:`\\theta \\leftarrow \\mathbf{0}`
            at epoch 0).
        device: Device used for the forward pass.
        predicted: Cached ImageNet predictions (skips the forward pass).
        targets: Cached target labels (skips the forward pass).
        max_batches: Optional cap on the number of batches (debug/quick runs).
        ignore_index: Sentinel for unmatched pre-trained labels.
        return_frequency_matrix: Also return ``d`` under key ``"frequency_matrix"``.

    Returns:
        ``dict`` with ``target_to_pretrained``, ``pretrained_to_target`` and,
        optionally, ``frequency_matrix``.
    """
    d = compute_frequency_matrix(
        model,
        data_loader,
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        f_in=f_in,
        device=device,
        max_batches=max_batches,
        predicted=predicted,
        targets=targets,
    )
    mapping = ilm_mapping_from_frequency_matrix(
        d, num_target_classes=num_target_classes, ignore_index=ignore_index
    )
    if not return_frequency_matrix:
        mapping.pop("frequency_matrix", None)
    return mapping


def ilm_mapping_from_predictions(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    num_pretrained_classes: int,
    num_target_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Ilm update from cached predictions (useful for unit tests / fast epochs)."""
    d = frequency_matrix_from_predictions(
        predicted,
        targets,
        num_pretrained_classes=num_pretrained_classes,
        num_target_classes=num_target_classes,
    )
    return ilm_mapping_from_frequency_matrix(
        d, num_target_classes=num_target_classes, ignore_index=ignore_index
    )


# ---------------------------------------------------------------------------
# Module wrapper: f_out^{Ilm}
# ---------------------------------------------------------------------------
class IlmMapping(nn.Module):
    """:math:`f_{\\mathrm{out}}^{\\mathrm{Ilm}}` — the epoch-wise iterated output mapping.

    The mapping itself is non-parametric (integer index buffers).  Its single extra
    behaviour compared to :class:`~smm_vr.label_mapping.flm.FlmMapping` is
    :attr:`recomputes_each_epoch = True`: the training loop (Algorithm 1) is expected
    to call :meth:`update` before optimising each epoch, which refreshes the mapping
    from a freshly computed frequency matrix using the current
    :math:`f_{\\mathrm{in}}` (Eq. (11)).

    Args:
        target_to_pretrained: :math:`(|\\mathcal{Y}^{\\mathrm{T}}|,)` long tensor where
            entry ``t`` is the ImageNet logit index used for target class ``t``.
        pretrained_to_target: Optional :math:`(|\\mathcal{Y}^{\\mathrm{P}}|,)` long tensor
            giving the literal ``f_out`` view (``IGNORE_INDEX`` where unmatched).
        num_pretrained_classes: Size of the ImageNet label space; inferred when omitted.
        ignore_index: Sentinel for unmatched pre-trained labels.
    """

    #: Tells the training loop to refresh this mapping before every epoch.
    recomputes_each_epoch: bool = True
    #: Human-readable identifier used in reports/configs.
    name: str = "Ilm"

    def __init__(
        self,
        target_to_pretrained: torch.Tensor,
        pretrained_to_target: Optional[torch.Tensor] = None,
        num_pretrained_classes: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        target_to_pretrained = torch.as_tensor(target_to_pretrained, dtype=torch.long)
        if target_to_pretrained.dim() != 1:
            raise ValueError(
                "target_to_pretrained must be a 1-D tensor of ImageNet indices, "
                f"got shape {tuple(target_to_pretrained.shape)}"
            )

        if pretrained_to_target is None:
            if num_pretrained_classes is None:
                num_pretrained_classes = (
                    int(target_to_pretrained.max().item()) + 1
                    if target_to_pretrained.numel()
                    else 0
                )
            pretrained_to_target = torch.full(
                (int(num_pretrained_classes),), ignore_index, dtype=torch.long
            )
            valid = (
                (target_to_pretrained >= 0)
                & (target_to_pretrained < int(num_pretrained_classes))
            )
            if valid.any():
                pretrained_to_target[
                    target_to_pretrained[valid]
                ] = torch.arange(target_to_pretrained.numel(), dtype=torch.long)[valid]
        else:
            pretrained_to_target = torch.as_tensor(
                pretrained_to_target, dtype=torch.long
            )
            if num_pretrained_classes is None:
                num_pretrained_classes = int(pretrained_to_target.numel())

        self.ignore_index = int(ignore_index)
        self.register_buffer("target_to_pretrained", target_to_pretrained)
        self.register_buffer("pretrained_to_target", pretrained_to_target)
        # Metadata (not tensors): preserves the state across `update` calls.
        self._num_pretrained_classes = int(num_pretrained_classes)
        self.num_updates = 0
        self.converged = False
        self.history: list = []

    # -- properties ---------------------------------------------------------
    @property
    def num_target_classes(self) -> int:
        """:math:`|\\mathcal{Y}^{\\mathrm{T}}|` (number of mapped target classes)."""
        return int(self.target_to_pretrained.numel())

    @property
    def num_pretrained_classes(self) -> int:
        """:math:`|\\mathcal{Y}^{\\mathrm{P}}|` (size of the ImageNet label space)."""
        return int(self._num_pretrained_classes)

    # -- forward ------------------------------------------------------------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Map ImageNet logits to target-class logits (Eq. (1) / §2.3).

        Args:
            logits: ``(..., |\\mathcal{Y}^{\\mathrm{P}}|)`` tensor of pre-trained logits.

        Returns:
            ``(..., |\\mathcal{Y}^{\\mathrm{T}}|)`` tensor of matched logits, ordered by
            target class (so class index ``t`` is trained against label ``t``).
        """
        return apply_label_mapping(logits, self.target_to_pretrained)

    # -- mapping refresh (Algorithm 4) -------------------------------------
    def set_mapping(self, mapping: Dict[str, torch.Tensor]) -> None:
        """Install a mapping dict produced by the functions above (no recomputation)."""
        old = self.target_to_pretrained.detach().clone()
        new = torch.as_tensor(mapping["target_to_pretrained"], dtype=torch.long)
        if new.numel() != old.numel():
            raise ValueError(
                "mapping size mismatch: expected "
                f"{old.numel()} target classes, got {new.numel()}"
            )
        p2t = mapping.get("pretrained_to_target")
        if p2t is None:
            p2t = torch.full(
                (self.num_pretrained_classes,), self.ignore_index, dtype=torch.long
            )
            valid = (new >= 0) & (new < self.num_pretrained_classes)
            if valid.any():
                p2t[new[valid]] = torch.arange(new.numel(), dtype=torch.long)[valid]
        p2t = torch.as_tensor(p2t, dtype=torch.long)
        if p2t.numel() > self.pretrained_to_target.numel():
            self._num_pretrained_classes = int(p2t.numel())
        # Keep the buffer sizes consistent even if the label space grew.
        if p2t.numel() != self.pretrained_to_target.numel():
            self.register_buffer("pretrained_to_target", p2t)
        else:
            self.pretrained_to_target.copy_(p2t)
        self.target_to_pretrained.copy_(new)

        self.converged = bool(torch.equal(old, new))
        if not self.converged:
            self.history.append(new.clone())
            if len(self.history) > 50:  # keep the log bounded
                self.history = self.history[-25:]
        self.num_updates += 1

    def update(
        self,
        model: Optional[nn.Module] = None,
        data_loader=None,
        num_target_classes: Optional[int] = None,
        num_pretrained_classes: Optional[int] = None,
        f_in=None,
        device: Optional[torch.device] = None,
        *,
        frequency_matrix: Optional[torch.Tensor] = None,
        predicted: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        max_batches: Optional[int] = None,
        ignore_index: Optional[int] = None,
        max_iterations: int = 1,
    ) -> bool:
        """Refresh :math:`f_{\\mathrm{out}}^{\\mathrm{Ilm}}` (Algorithm 4).

        Called by the Algorithm 1 training loop *before* optimising each epoch.  Three
        ways to supply the statistics, in decreasing priority:

        1. ``frequency_matrix``: a precomputed :math:`d` (Algorithm 2 output);
        2. ``predicted`` / ``targets``: cached ImageNet predictions and target labels;
        3. ``model`` + ``data_loader`` (+ optional current ``f_in``): compute :math:`d`
           on the fly with the *current* reprogramming parameters, per Eq. (11).

        Note:
            Ilm is specified to update *each epoch* (Algorithm 4).  ``max_iterations``
            mirrors the paper's "iteratively until convergence" wording by allowing
            extra passes until the greedy mapping stops changing; with the default
            ``max_iterations=1`` the behaviour is exactly one refresh per epoch.

        Returns:
            ``True`` if the mapping is unchanged (converged), else ``False``.
        """
        num_target_classes = num_target_classes or self.num_target_classes
        num_pretrained_classes = (
            num_pretrained_classes or self._num_pretrained_classes
        )
        ignore_index = self.ignore_index if ignore_index is None else int(ignore_index)

        converged = False
        for _ in range(max(1, int(max_iterations))):
            if frequency_matrix is not None:
                mapping = ilm_mapping_from_frequency_matrix(
                    frequency_matrix,
                    num_target_classes=num_target_classes,
                    ignore_index=ignore_index,
                )
            elif predicted is not None and targets is not None:
                mapping = ilm_mapping_from_predictions(
                    predicted,
                    targets,
                    num_pretrained_classes=int(num_pretrained_classes),
                    num_target_classes=int(num_target_classes),
                    ignore_index=ignore_index,
                )
            else:
                if model is None or data_loader is None:
                    raise ValueError(
                        "IlmMapping.update requires either `frequency_matrix`, "
                        "`predicted`+`targets`, or `model`+`data_loader`."
                    )
                mapping = compute_ilm_mapping(
                    model,
                    data_loader,
                    num_target_classes=int(num_target_classes),
                    num_pretrained_classes=int(num_pretrained_classes),
                    f_in=f_in,
                    device=device,
                    max_batches=max_batches,
                    ignore_index=ignore_index,
                    return_frequency_matrix=True,
                )
                frequency_matrix = mapping.get("frequency_matrix")

            self.set_mapping(mapping)
            converged = self.converged
            if converged and frequency_matrix is None:
                break

        return converged

    # -- helpers ------------------------------------------------------------
    def map_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the arg-max target class predicted for each sample (or ``IGNORE_INDEX``)."""
        mapped = self.forward(logits)
        return mapped.argmax(dim=-1)

    def is_injective(self) -> bool:
        """Check the §2.3 requirement :math:`f_{\\mathrm{out}}(y_1) \\neq f_{\\mathrm{out}}(y_2)`."""
        idx = self.target_to_pretrained
        if idx.numel() <= 1:
            return True
        return bool(torch.unique(idx).numel() == idx.numel())

    def validate(self) -> None:
        """Raise ``ValueError`` if the stored mapping violates injectivity."""
        if not self.is_injective():
            raise ValueError(
                "Ilm mapping is not injective: "
                f"{self.target_to_pretrained.tolist()}"
            )

    def extra_repr(self) -> str:
        return (
            f"num_target_classes={self.num_target_classes}, "
            f"num_pretrained_classes={self.num_pretrained_classes}, "
            f"updates={self.num_updates}, injective={self.is_injective()}, "
            f"recomputes_each_epoch={self.recomputes_each_epoch}"
        )


#: Paper-style name for :math:`f_{\\mathrm{out}}^{\\mathrm{Ilm}}`.
IlmLabelMapping = IlmMapping


def update_ilm_mapping(
    mapping: IlmMapping,
    model: Optional[nn.Module] = None,
    data_loader=None,
    f_in=None,
    device: Optional[torch.device] = None,
    *,
    frequency_matrix: Optional[torch.Tensor] = None,
    predicted: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    num_target_classes: Optional[int] = None,
    num_pretrained_classes: Optional[int] = None,
) -> bool:
    """Functional shortcut for the per-epoch Ilm refresh (Algorithm 1 + Algorithm 4)."""
    return mapping.update(
        model=model,
        data_loader=data_loader,
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        f_in=f_in,
        device=device,
        frequency_matrix=frequency_matrix,
        predicted=predicted,
        targets=targets,
        max_batches=max_batches,
    )


def build_ilm_mapping(
    model: nn.Module,
    data_loader,
    num_target_classes: int,
    num_pretrained_classes: Optional[int] = None,
    f_in=None,
    device: Optional[torch.device] = None,
    *,
    predicted: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
) -> IlmMapping:
    """Build the default (Table 1/2) output mapping: Algorithm 2 + Algorithm 3 + Algorithm 4.

    Runs one initial Ilm update with the current :math:`f_{\\mathrm{in}}` (at epoch 0
    this is the zero-initialised :math:`\\delta`, so the result coincides with Flm's
    initialisation) and returns a ready-to-use, epoch-updatable module.
    """
    mapping = compute_ilm_mapping(
        model,
        data_loader,
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        f_in=f_in,
        device=device,
        predicted=predicted,
        targets=targets,
        max_batches=max_batches,
        ignore_index=ignore_index,
        return_frequency_matrix=False,
    )
    module = IlmMapping(
        target_to_pretrained=mapping["target_to_pretrained"],
        pretrained_to_target=mapping.get("pretrained_to_target"),
        num_pretrained_classes=num_pretrained_classes,
        ignore_index=ignore_index,
    )
    return module
