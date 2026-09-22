"""The APT adapter (Section 4.1 of the paper).

The APT adapter is built "over LoRA" but additionally carries

* an *input* pruning mask  ``m_i``  (size ``d_i``) and
* an *output* pruning mask ``m_o``  (size ``d_o``),

so that a linear operator ``H = W X`` becomes

    H_apt(X) = m_o . (W + s * W_B W_A) X . m_i                    (paper Eq. 2)

where ``.`` is a Hadamard product (broadcast over the batch / sequence axes),
``s`` is a constant scaling factor (the paper uses ``s = 2``), and the rank
``r_apt`` of ``W_A in R^{r_apt x d_i}`` / ``W_B in R^{d_o x r_apt}`` is
dynamically increased during fine-tuning (Section 4.3).

Mask semantics
--------------
* ``m_i`` is indexed by the *input* feature.  Zeroing ``m_i[j]`` removes column
  ``j`` of ``W`` (and of ``W_A``), i.e. it prunes the ``j``-th input dimension.
* ``m_o`` is indexed by the *output* feature.  Zeroing ``m_o[i]`` removes row
  ``i`` of ``W`` (and of ``W_B``), i.e. it prunes the ``i``-th output unit.

In MHA layers ``m_o`` prunes attention heads and in FFN layers it prunes
intermediate neurons, while ``m_i`` always prunes the transformer hidden
dimension.  See ``apt.blocks`` for how physical weight slices are grouped into
prunable blocks.

Activation / gradient bookkeeping
---------------------------------
The adapter accumulates two compressed statistics per feature (Algorithm 1 of
the paper)::

    <|activation|> summed over batch & sequence
    <|gradient|>   summed over batch & sequence

Their product is the lightweight, "compressed activation-gradient product"
salience estimate used by APT (Appendix B).  The statistics are only collected
while ``track`` is enabled so that inference is completely overhead free.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RequireInputGrad(torch.autograd.Function):
    """Identity that guarantees its input requires grad.

    Needed so that the very first APT adapter of a network (whose input is a
    *frozen* embedding output) can still receive ``dL/dX``.  When the input
    already requires grad the tensor is returned unchanged and the graph is
    left intact.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.requires_grad:
            return x
        return x.detach().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output


class APTLinear(nn.Module):
    """A frozen :class:`torch.nn.Linear` augmented with an APT adapter.

    Parameters
    ----------
    base:
        The pretrained linear layer.  Its parameters are frozen in place.
    r:
        Initial APT adapter rank ``r_apt`` (the paper initialises it to 8).
    scaling:
        Constant LoRA scaling factor ``s`` (the paper uses 2 for all layers).
    use_lora:
        ``True`` for the query / value (and small-model FFN) layers that carry
        trainable APT adapter weights.  ``False`` for frozen-only layers
        (e.g. the key / output projection): those layers still get pruning
        masks and salience tracking but contribute no tuning parameters.
    track:
        Whether to accumulate salience statistics while in ``.train()`` mode.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        scaling: float = 2.0,
        use_lora: bool = True,
        track: bool = True,
        dropout: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype
        self.use_lora = use_lora
        self.scaling = float(scaling)
        self.track = track
        #: whether to accumulate the streaming moments needed for the kurtosis
        #: term of the outlier-aware salience (Eq. 5).  The trainer keeps this
        #: enabled during the pruning stage only.
        self.track_kurtosis = True
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- pruning masks (kept as buffers so they move with the module) ----
        self.register_buffer("mask_in", torch.ones(self.in_features, device=dev, dtype=dt), persistent=True)
        self.register_buffer("mask_out", torch.ones(self.out_features, device=dev, dtype=dt), persistent=True)

        # --- low rank tuning parameters -------------------------------------
        self.rank = int(r)
        if use_lora:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features, device=dev, dtype=dt))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=dev, dtype=dt))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # --- salience accumulators (CPU float64 for numerical stability) ----
        self._acc_in_act: Optional[torch.Tensor] = None
        self._acc_out_act: Optional[torch.Tensor] = None
        self._acc_in_grad: Optional[torch.Tensor] = None
        self._acc_out_grad: Optional[torch.Tensor] = None
        self._mom_in: Optional[torch.Tensor] = None
        self._mom_out: Optional[torch.Tensor] = None
        self._mom_count = 0
        self._n_forward = 0
        self._hooks_installed = False
        if track:
            self.install_hooks()

    # ---------------------------------------------------- nn.Linear interface
    # Some model implementations (e.g. ``T5DenseActDense``) introspect
    # ``module.weight`` / ``module.bias`` directly, so the wrapper exposes the
    # frozen parameters under the usual names.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @weight.setter
    def weight(self, value) -> None:
        self.base.weight = nn.Parameter(value)

    @property
    def bias(self):
        return self.base.bias

    @bias.setter
    def bias(self, value) -> None:
        self.base.bias = None if value is None else nn.Parameter(value)

    # ------------------------------------------------------------------ init
    def reset_lora_parameters(self) -> None:
        """Standard LoRA initialisation: ``A ~ kaiming``, ``B = 0``.

        Because ``B`` is zero the adapter output is exactly zero before
        training, so the wrapped LM behaves identically to the pretrained one.
        """
        if not self.use_lora:
            return
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # ----------------------------------------------------------------- hooks
    def install_hooks(self) -> None:
        if self._hooks_installed or not self.track:
            return
        self.register_full_backward_hook(self._backward_hook)
        self._hooks_installed = True

    def _backward_hook(self, module, grad_input, grad_output):  # noqa: D401
        if not self.training:
            return None
        if grad_input and grad_input[0] is not None:
            self._accumulate_grad("in", grad_input[0])
        if grad_output and grad_output[0] is not None:
            self._accumulate_grad("out", grad_output[0])
        return None

    def _accumulate_grad(self, which: str, g: torch.Tensor) -> None:
        with torch.no_grad():
            flat = g.detach().reshape(-1, g.shape[-1]).abs().sum(dim=0)
            # accumulate on the CPU in float64 (MPS/CUDA do not all support f64)
            flat = flat.cpu().double()
            if which == "in":
                self._acc_in_grad = flat if self._acc_in_grad is None else self._acc_in_grad + flat
            else:
                self._acc_out_grad = flat if self._acc_out_grad is None else self._acc_out_grad + flat

    def _accumulate_act(self, which: str, x: torch.Tensor) -> None:
        with torch.no_grad():
            flat = x.detach().reshape(-1, x.shape[-1]).abs().sum(dim=0)
            flat = flat.cpu().double()
            if which == "in":
                self._acc_in_act = flat if self._acc_in_act is None else self._acc_in_act + flat
            else:
                self._acc_out_act = flat if self._acc_out_act is None else self._acc_out_act + flat

    def _accumulate_moments(self, which: str, x: torch.Tensor) -> None:
        """Streaming raw moments ``sum x, sum x^2, sum x^3, sum x^4`` per feature."""
        with torch.no_grad():
            v = x.detach().reshape(-1, x.shape[-1])
            m = torch.stack([v.sum(0), v.pow(2).sum(0), v.pow(3).sum(0), v.pow(4).sum(0)])
            m = m.cpu().double()
            if which == "in":
                self._mom_in = m if self._mom_in is None else self._mom_in + m
            else:
                self._mom_out = m if self._mom_out is None else self._mom_out + m
        if which == "out":
            self._mom_count += int(v.shape[0])

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tracking = self.training and self.track
        if tracking:
            x = _RequireInputGrad.apply(x)
            self._accumulate_act("in", x)
            if self.track_kurtosis:
                self._accumulate_moments("in", x)

        x_masked = x * self.mask_in
        out = self.base(x_masked)

        if self.use_lora:
            lora = F.linear(self.dropout(x_masked), self.lora_A)
            out = out + self.scaling * F.linear(lora, self.lora_B)

        if tracking:
            self._accumulate_act("out", out)
            if self.track_kurtosis:
                self._accumulate_moments("out", out)
        return out * self.mask_out

    # ---------------------------------------------------------------- merged
    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        """``m_o . (W + s W_B W_A) . m_i`` with the masks applied to the weights."""
        w = self.base.weight.detach().clone()
        if self.use_lora:
            w = w + self.scaling * (self.lora_B.detach() @ self.lora_A.detach())
        w = w * self.mask_out.unsqueeze(1) * self.mask_in.unsqueeze(0)
        return w

    @torch.no_grad()
    def merge_lora(self) -> None:
        """Fold the (trained) adapter weights into the frozen weight.

        Used right before inference so that the adapter costs nothing at
        runtime -- mirroring the paper's statement that "tuning parameters ...
        can be fully merged after training".
        """
        if not self.use_lora:
            return
        self.base.weight.add_(self.scaling * (self.lora_B @ self.lora_A))
        # the low-rank branch is now folded into ``W``; zero it so that the
        # forward pass stays numerically identical
        self.lora_A.zero_()
        self.lora_B.zero_()

    # ------------------------------------------------------------ rank growth
    @torch.no_grad()
    def grow_rank(self, new_rank: int) -> None:
        """Increase ``r_apt`` while keeping the layer output unchanged.

        Following the paper (Section 4.3) the new entries of ``W_A`` are drawn
        from ``N(0, sigma^2)`` and ``W_B`` is zero padded, which is exactly the
        LoRA initialisation and therefore leaves ``W_B W_A`` untouched.
        """
        if not self.use_lora or new_rank <= self.rank:
            return
        device = self.lora_A.device
        extra = new_rank - self.rank
        new_A = torch.zeros(new_rank, self.in_features, device=device, dtype=self.lora_A.dtype)
        new_A[: self.rank] = self.lora_A.data
        nn.init.kaiming_uniform_(new_A[self.rank:], a=math.sqrt(5))

        new_B = torch.zeros(self.out_features, new_rank, device=device, dtype=self.lora_B.dtype)
        new_B[:, : self.rank] = self.lora_B.data

        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.rank = int(new_rank)
        _ = extra  # documented above

    # ------------------------------------------------------- salience pieces
    def reset_statistics(self) -> None:
        self._acc_in_act = None
        self._acc_out_act = None
        self._acc_in_grad = None
        self._acc_out_grad = None
        self._mom_in = None
        self._mom_out = None
        self._mom_count = 0
        self._n_forward = 0

    def activation_kurtosis(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``(kurt_in, kurt_out)`` of the activation distributions of this layer."""
        from .blocks import kurtosis_from_moments

        kin = None
        kout = None
        if self._mom_in is not None and self._mom_count > 0:
            kin = kurtosis_from_moments(self._mom_in, self._mom_count)
        if self._mom_out is not None and self._mom_count > 0:
            kout = kurtosis_from_moments(self._mom_out, self._mom_count)
        return kin, kout

    def compressed_salience(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compressed activation x gradient salience of the input / output dims.

        Returns ``(sal_in, sal_out)`` where::

            sal_in[j]  = (sum |dL/dX_j|) * (sum |X_j|)
            sal_out[i] = (sum |dL/dH_i|) * (sum |H_i|)

        matching Equation (5) / Algorithm 1 of the paper.
        """
        sal_in = None
        sal_out = None
        if self._acc_in_act is not None and self._acc_in_grad is not None:
            sal_in = self._acc_in_act * self._acc_in_grad
        if self._acc_out_act is not None and self._acc_out_grad is not None:
            sal_out = self._acc_out_act * self._acc_out_grad
        return sal_in, sal_out

    def tuning_salience(self) -> Optional[torch.Tensor]:
        """Per-output-dimension tuning salience ``sum_q |W_B[i,q] dL/dW_B[i,q]|``.

        Appendix B: the real block salience of a LoRA layer is the frozen
        weight salience *plus* the salience of the corresponding tuning weights,
        so we return the tuning contribution separately for the caller to add.
        """
        if not self.use_lora or self.lora_B.grad is None:
            return None
        return (
            (self.lora_B.detach().abs() * self.lora_B.grad.detach().abs())
            .sum(dim=1)
            .cpu()
            .double()
        )

    def tuning_salience_in(self) -> Optional[torch.Tensor]:
        """Per-input-dimension tuning salience ``s * sum_q |W_A[q,j] dL/dW_A[q,j]|``."""
        if not self.use_lora or self.lora_A.grad is None:
            return None
        return (
            (self.lora_A.detach().abs() * self.lora_A.grad.detach().abs())
            .sum(dim=0)
            .cpu()
            .double()
        )

    def adapter_importance(self) -> float:
        """``I(H_apt) = sum_{i,j} |W_B[i,j] * dL/dW_B[i,j]|`` (Section 4.3)."""
        if not self.use_lora or self.lora_B.grad is None:
            return 0.0
        return float(
            (self.lora_B.detach().abs() * self.lora_B.grad.detach().abs()).sum().item()
        )

    # ------------------------------------------------------------------ misc
    def n_tuning_parameters(self) -> int:
        if not self.use_lora:
            return 0
        return self.rank * (self.in_features + self.out_features)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.rank}, "
            f"scaling={self.scaling}, lora={self.use_lora}"
        )
