"""The two adversarial fine-tuning objectives of the paper.

FARE (Eq. (3), Sec. 3.3) -- *unsupervised*::

    L_FARE(phi, x) = max_{||z - x||_inf <= eps} || phi(z) - phi_org(x) ||_2^2

The reference embedding ``phi_org(x)`` is produced by the **original, frozen**
CLIP encoder on the **clean** image; only ``phi(z)`` (the fine-tuned encoder
evaluated at the perturbed image) contributes gradients.  Because
``L_FARE -> 0`` implies ``phi(x) -> phi_org(x)`` on clean inputs, the fine-tuned
encoder can be dropped into LLaVA / OpenFlamingo without any retraining: this is
what Theorem 3.1 makes precise (the embedding distance bounds the change of the
cosine similarities, hence of the zero-shot logits).

TeCoA (Eq. (2), Sec. 3.2) -- *supervised* baseline of Mao et al. (2023)::

    L_TeCoA(y, f(phi, x)) = -log softmax_y( cos(phi(x), psi(t_k)) )_k

which is adversarial training on the fixed set of ImageNet text embeddings
``psi(t_k)``.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .pgd import pgd_linf_maximize


EncodeFn = Callable[[torch.Tensor], torch.Tensor]


class _AdversarialLossBase:
    """Shared plumbing: inner maximisation + the outer (minimised) loss."""

    def __init__(
        self,
        eps: float,
        alpha: float,
        n_steps: int = 10,
        random_init: bool = True,
        momentum: float = 0.9,
        grad_normalization: str = "elementwise_sign",
        keep_best: bool = False,
        quantize_bits: Optional[int] = None,
    ):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.n_steps = int(n_steps)
        self.random_init = random_init
        self.momentum = momentum
        self.grad_normalization = grad_normalization
        self.keep_best = keep_best
        self.quantize_bits = quantize_bits

    def inner_max(self, objective: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor):
        return pgd_linf_maximize(
            objective,
            x,
            eps=self.eps,
            alpha=self.alpha,
            n_steps=self.n_steps,
            random_init=self.random_init,
            momentum=self.momentum,
            grad_normalization=self.grad_normalization,
            quantize_bits=self.quantize_bits,
            keep_best=self.keep_best,
        )


class FARELoss(_AdversarialLossBase):
    """Unsupervised embedding-preserving adversarial fine-tuning loss.

    Parameters
    ----------
    encode_fn:
        callable mapping a pixel-space batch to the embedding of the model that
        is being fine-tuned, i.e. :math:`\\phi(z)`.
    reference_encoder:
        the *original* CLIP image encoder used to compute the fixed targets
        ``phi_org(x)``.  It is never updated and can live on the same device as
        the fine-tuned encoder.
    feature:
        ``"projected_class_token"`` (default) or ``"class_token"``; see
        :class:`robust_clip.models.CLIPImageEncoder`.  The paper computes the
        FARE loss with respect to the class token only (App. B.1).
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        reference_encoder,
        eps: float = 2 / 255,
        alpha: float = 1 / 255,
        n_steps: int = 10,
        **kwargs,
    ):
        super().__init__(eps=eps, alpha=alpha, n_steps=n_steps, **kwargs)
        self.encode = encode_fn
        self.reference_encoder = reference_encoder
        for param in self.reference_encoder.parameters():
            param.requires_grad_(False)
        self.reference_encoder.eval()

    @torch.no_grad()
    def reference_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """``phi_org(x)`` for the *clean* images, detached from the graph."""
        was_training = self.reference_encoder.training
        self.reference_encoder.eval()
        emb = self.reference_encoder.encode_image(x)
        if was_training:
            self.reference_encoder.train()
        return emb.detach()

    def embedding_distance(self, x_adv: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Per-sample ``|| phi(z) - phi_org(x) ||_2^2`` (the outer loss)."""
        z = self.encode(x_adv)
        return (z - reference).pow(2).flatten(1).sum(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, x_adv)``.

        ``loss`` is the *outer* FARE loss (mean over the batch) evaluated at the
        adversarial point, and it is differentiable w.r.t. the encoder behind
        ``encode_fn``.
        """
        reference = self.reference_embedding(x)

        def objective(x_adv: torch.Tensor) -> torch.Tensor:
            return self.embedding_distance(x_adv, reference)

        x_adv, _ = self.inner_max(objective, x)
        loss = self.embedding_distance(x_adv.detach(), reference).mean()
        return loss, x_adv.detach()

    __call__ = forward


class TeCoALoss(_AdversarialLossBase):
    """Supervised adversarial fine-tuning on the ImageNet zero-shot classifier.

    The logits are the cosine similarities between the (fine-tuned) image
    embedding and the fixed text embeddings of the ImageNet classes,
    ``f_k(phi, x) = cos(phi(x), psi(t_k))``.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        text_embeddings: torch.Tensor,
        eps: float = 2 / 255,
        alpha: float = 1 / 255,
        n_steps: int = 10,
        use_logit_scale: bool = False,
        temperature: float = 100.0,
        **kwargs,
    ):
        super().__init__(eps=eps, alpha=alpha, n_steps=n_steps, **kwargs)
        self.encode = encode_fn
        self.text_embeddings = F.normalize(text_embeddings.float().detach(), dim=-1).clone()
        self.temperature = float(temperature)

    def logits(self, x_adv: torch.Tensor) -> torch.Tensor:
        """``f_k(phi, x) = cos(phi(x), psi(t_k))`` scaled by the CLIP logit scale."""
        z = F.normalize(self.encode(x_adv).float(), dim=-1)
        text = self.text_embeddings.to(z.device, z.dtype)
        return self.temperature * (z @ text.t())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def objective(x_adv: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(self.logits(x_adv), y, reduction="none")

        x_adv, _ = self.inner_max(objective, x)
        loss = F.cross_entropy(self.logits(x_adv.detach()), y)
        return loss, x_adv.detach()

    __call__ = forward


def build_loss(method: str, **kwargs):
    method = method.lower()
    if method in ("fare", "unsupervised"):
        return FARELoss(**kwargs)
    if method in ("tecoa", "supervised"):
        return TeCoALoss(**kwargs)
    raise ValueError(f"unknown fine-tuning method '{method}'")
