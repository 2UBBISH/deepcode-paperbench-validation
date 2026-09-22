"""Projected Gradient Descent (l_inf) attack, implemented exactly to the
Addendum specification of the Robust CLIP benchmark.

Addendum requirements implemented here
--------------------------------------
1. ``gradient normalization with elementwise sign for l_infinity`` -- the
   gradient is normalized (by its mean absolute value over all non-batch
   dimensions) and then an elementwise ``sign`` is taken.
2. ``momentum factor of 0.9`` -- an exponential moving average of the
   (normalized, signed) gradients is accumulated with ``mu = 0.9``.
3. ``initialization with uniform random perturbation`` -- the perturbation is
   initialized uniformly inside the l_inf ball, i.e. ``delta ~ U(-eps, eps)``.
4. ``computation of l_infinity ball around non-normalized inputs`` -- the
   projection is done on the raw pixel tensors *before* any model-specific
   normalization is applied.

The Addendum does **not** state ``eps``, ``alpha`` or the iteration count for
this general PGD; those are therefore taken from configuration with no invented
"paper" values (see ``configs/pgd_eval.yaml``, where they are marked as
``unspecified``).

Precision policy (Addendum)
---------------------------
Half-precision attacks store perturbations as ``int16``; single-precision
attacks store them as ``int32``.  The perturbation is kept in the mandated
integer dtype when it is not being used in a differentiable computation, via
``utils.precision``.

The interface is intentionally shared with :mod:`robust_clip_repro.attacks.apgd`
so that evaluation harnesses can switch attackers transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import torch

from ..utils.precision import (
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
)

logger = logging.getLogger(__name__)

#: Momentum factor mandated by the Addendum.
MOMENTUM = 0.9


@dataclass
class PGDLinfConfig:
    """Configuration for the l_inf PGD attack.

    ``eps``/``alpha``/``iterations`` are *not* specified by the Addendum; they
    must be supplied externally (paper body lookup or user config).  Defaults
    here are conservative placeholders documented as unspecified.
    """

    eps: Optional[float] = None            # UNSPECIFIED by the Addendum
    alpha: Optional[float] = None          # UNSPECIFIED by the Addendum
    iterations: Optional[int] = None       # UNSPECIFIED by the Addendum
    momentum: float = MOMENTUM             # Addendum: 0.9
    random_start: bool = True              # Addendum: uniform random perturbation
    targeted: bool = False
    #: Scaling used when converting the (already eps-scaled) perturbation to and
    #: from the mandated integer storage dtype.
    quant_scale: float = 1.0

    def resolve(self, eps: float, alpha: float, iterations: int) -> "PGDLinfConfig":
        """Return a copy with the externally supplied budget filled in."""
        return PGDLinfConfig(
            eps=eps,
            alpha=alpha,
            iterations=iterations,
            momentum=self.momentum,
            random_start=self.random_start,
            targeted=self.targeted,
            quant_scale=self.quant_scale,
        )


@dataclass
class PGDLinfAttack:
    """Multi-step l_inf PGD with momentum.

    Parameters
    ----------
    eps:
        l_inf radius of the ball, in **raw pixel space** (non-normalized inputs).
    alpha:
        Step size, in raw pixel space.
    iterations:
        Number of PGD steps.
    precision:
        ``"half"`` or ``"single"``.  Selects the mandated integer storage dtype
        (int16 / int32).
    targeted:
        If ``True`` a targeted attack (maximize the loss w.r.t. ``y_target``)
        is performed; otherwise the untargeted variant (maximize the loss
        w.r.t. the true label) is used.
    momentum:
        Momentum factor (Addendum: 0.9).
    random_start:
        Uniformly initialize the perturbation inside the l_inf ball.
    """

    eps: float
    alpha: float
    iterations: int
    precision: str = "single"
    targeted: bool = False
    momentum: float = MOMENTUM
    random_start: bool = True
    quant_scale: float = 1.0
    clamp: Optional[Tuple[float, float]] = None
    config: PGDLinfConfig = field(default_factory=PGDLinfConfig)

    def __post_init__(self) -> None:
        if self.eps is None or self.alpha is None or self.iterations is None:
            raise ValueError(
                "eps, alpha and iterations are not specified by the Addendum; "
                "supply them explicitly via config."
            )
        self.float_dtype = float_dtype_for_precision(self.precision)
        # Addendum precision policy: int16 for half-precision, int32 for single.
        self.int_dtype = int_dtype_for_precision(self.precision)
        self.config = PGDLinfConfig(
            eps=self.eps,
            alpha=self.alpha,
            iterations=self.iterations,
            momentum=self.momentum,
            random_start=self.random_start,
            targeted=self.targeted,
            quant_scale=self.quant_scale,
        )
        # Sanity: momentum mandated to be 0.9 for the general PGD.
        assert float(self.momentum) == MOMENTUM, "Addendum mandates momentum 0.9"

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def initial_perturbation(
        self, x: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Uniform random perturbation initialization inside the l_inf ball.

        The ball is computed around the non-normalized input ``x``.  The
        returned tensor is stored in the mandated integer dtype.
        """
        fdt = self.float_dtype
        x_f = x.to(fdt)
        if not self.random_start:
            delta = torch.zeros_like(x_f)
        else:
            delta = torch.zeros_like(x_f).uniform_(-self.eps, self.eps, generator=generator)
        return encode_perturbation(delta, self.precision, quant_scale=self.quant_scale)

    def project(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Project ``delta`` onto the l_inf ball around the raw input ``x``.

        ``x`` is the *non-normalized* (raw pixel) input tensor.  If ``clamp``
        is given, the adversarial example is additionally clipped to that
        range, and the perturbation is recomputed accordingly.
        """
        fdt = self.float_dtype
        delta_f = delta.to(fdt)
        delta_f = delta_f.clamp(-self.eps, self.eps)
        if self.clamp is not None:
            lo, hi = self.clamp
            delta_f = (x.to(fdt) + delta_f).clamp(lo, hi) - x.to(fdt)
            delta_f = delta_f.clamp(-self.eps, self.eps)
        return encode_perturbation(delta_f, self.precision, quant_scale=self.quant_scale)

    @staticmethod
    def normalize_and_sign(grad: torch.Tensor) -> torch.Tensor:
        """Gradient normalization followed by elementwise sign (l_inf).

        The gradient is divided by its mean absolute value computed over all
        dimensions except the batch dimension, then the elementwise ``sign`` is
        taken.  This is the standard l_inf gradient normalization used by the
        benchmark.
        """
        if grad.ndim > 1:
            dims = tuple(range(1, grad.ndim))
            grad = grad / (grad.abs().mean(dim=dims, keepdim=True) + 1e-12)
        return grad.sign()

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------
    def perturb(
        self,
        x: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        y_target: Optional[torch.Tensor] = None,
        label_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        return_delta: bool = True,
    ) -> torch.Tensor:
        """Run PGD against ``loss_fn``.

        Parameters
        ----------
        x:
            Raw (non-normalized) input batch.  The l_inf ball is computed
            around this tensor.
        loss_fn:
            Callable mapping a *model-space* input tensor to a scalar loss.
            It is expected to apply the model's own normalization internally:
            this attack only ever concretizes adversarial pixels as
            ``x + delta`` in raw space.
        y_target:
            Target labels for the targeted variant.
        label_fn:
            Optional callable mapping the current raw adversarial batch to a
            label tensor used by ``loss_fn`` (when the loss needs the labels).
            If omitted and the attack is untargeted, ``loss_fn`` alone decides.
        return_delta:
            If ``True`` return the integer-encoded perturbation ``delta``;
            otherwise return the raw adversarial pixels ``x + delta``.

        Returns
        -------
        torch.Tensor
            Either the int-encoded perturbation (mandated dtype) or the
            adversarial raw pixels.
        """
        if self.targeted and y_target is None:
            raise ValueError("targeted=True requires y_target")

        fdt = self.float_dtype
        x_f = x.to(fdt)
        delta = self.initial_perturbation(x, generator=generator)
        grad_momentum = torch.zeros_like(x_f)

        for step in range(int(self.iterations)):
            delta_f = decode_perturbation(
                delta, self.float_dtype, quant_scale=self.quant_scale
            )
            delta_f = delta_f.detach().requires_grad_(True)

            loss = self._compute_loss(loss_fn, x_f, delta_f, y_target, label_fn)
            grad = torch.autograd.grad(loss, delta_f, retain_graph=False)[0]

            # l_inf: normalize gradient then take elementwise sign.
            grad = self.normalize_and_sign(grad.detach())

            # Momentum accumulation (mu = 0.9).
            grad_momentum = self.momentum * grad_momentum + grad

            with torch.no_grad():
                delta_f = delta_f.detach() + self.alpha * grad_momentum.sign()
                # Projection onto the l_inf ball around NON-normalized inputs.
                delta = self.project(x_f, delta_f)

            if logger.isEnabledFor(logging.DEBUG) and (step % 50 == 0):
                d = decode_perturbation(delta, fdt, quant_scale=self.quant_scale)
                logger.debug(
                    "pgd step %d/%d  max|delta|=%.6f", step, self.iterations, float(d.abs().max())
                )

        delta_f = decode_perturbation(delta, fdt, quant_scale=self.quant_scale)
        delta = encode_perturbation(delta_f, self.precision, quant_scale=self.quant_scale)
        if return_delta:
            return delta
        return (x_f + delta_f).to(x.dtype)

    @staticmethod
    def _compute_loss(
        loss_fn: Callable[..., torch.Tensor],
        x_f: torch.Tensor,
        delta_f: torch.Tensor,
        y_target: Optional[torch.Tensor],
        label_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        x_adv = x_f + delta_f
        labels = label_fn(x_adv) if label_fn is not None else None
        try:
            if y_target is not None:
                return loss_fn(x_adv, y_target)
            if labels is not None:
                return loss_fn(x_adv, labels)
            return loss_fn(x_adv)
        except TypeError:
            # loss_fn is a single-argument callable.
            return loss_fn(x_adv)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def attack_untargeted(
        self,
        x: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        *,
        labels: Optional[torch.Tensor] = None,
        return_delta: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Untargeted PGD: maximize the loss w.r.t. the ground-truth labels."""
        return self.perturb(
            x,
            lambda xa, y=None: loss_fn(xa, labels if y is None else y),
            return_delta=return_delta,
            generator=generator,
        )

    def attack_targeted(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        *,
        return_delta: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Targeted PGD: *minimize* the loss w.r.t. the target (the caller is
        expected to pass e.g. a cross-entropy loss; we negate it internally)."""
        return self.perturb(
            x,
            lambda xa, y=None: -loss_fn(xa, y_target if y is None else y),
            y_target=y_target,
            return_delta=return_delta,
            generator=generator,
        )

    def adversarial_examples(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Concretize adversarial raw pixels from an int-encoded perturbation."""
        delta_f = decode_perturbation(
            delta, self.float_dtype, quant_scale=self.quant_scale
        )
        return (x.to(self.float_dtype) + delta_f).to(x.dtype)


def pgd_linf_attack(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    alpha: float,
    iterations: int,
    precision: str = "single",
    criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Functional helper mirroring the class API for quick experiments.

    ``model_fn`` maps raw pixels to logits (applying normalization internally).
    Returns the adversarial raw pixels.
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    attack = PGDLinfAttack(
        eps=eps,
        alpha=alpha,
        iterations=iterations,
        precision=precision,
        targeted=targeted,
        clamp=clamp,
    )

    def loss_fn(xa: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = model_fn(xa)
        if labels is None:
            labels = y
        return criterion(logits, labels)

    if targeted:
        delta = attack.attack_targeted(x, y_target, loss_fn, return_delta=True)
    else:
        delta = attack.attack_untargeted(x, loss_fn, labels=y, return_delta=True)
    return attack.adversarial_examples(x, delta)
