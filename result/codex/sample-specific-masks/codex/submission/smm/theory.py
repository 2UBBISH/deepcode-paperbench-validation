"""Theory of the paper: approximation error of the SMM hypothesis space.

Reproduced statements
---------------------
**Definition 4.1 (Approximation error).** For a distribution ``D`` over
``X x Y`` with Bayes risk ``R_D^*`` and a hypothesis space ``F = {f: X -> Y}``::

    Err_D^apx(F) = inf_{f in F} E_{(X,Y)~D}[ l(f(X), Y) ] - R_D^*

**Theorem 4.2.** If ``F1 subseteq F2`` then ``Err^apx(F1) >= Err^apx(F2)``
(proof: the infimum over a larger set cannot be larger; Appendix B.1).

**Hypothesis spaces.**
``F_shr(f_P') = { f | f(x) = f_P'(r(x) + M * delta) }``  (shared binary mask ``M``)
``F_sp(f_P')  = { f | f(x) = f_P'(r(x) + f_mask(r(x))) }`` (sample-specific pattern)
``F_smm(f_P') = { f | f(x) = f_P'(r(x) + f_mask(r(x)) * delta) }`` (SMM)

**Proposition 4.3.** ``F_shr(f_P') subseteq F_smm(f_P')`` and therefore
``Err^apx(F_shr) >= Err^apx(F_smm)``.  The proof (Appendix B.2) is constructive:
the last layer of ``f_mask`` is an affine convolution, so setting its weights to
zero and its bias to the binary mask ``M`` makes ``f_mask(r(x)) = M`` for every
``x``.  :func:`encode_shared_mask` performs this construction and
:func:`verify_shared_mask_inclusion` checks the resulting equality numerically.

**Proposition B.1.** ``F_sp(f_P') subseteq F_smm(f_P')`` because ``delta`` may
be the all-one matrix ``J``; see :func:`verify_sample_specific_inclusion`.

Finally, :func:`empirical_approximation_error_experiment` optimises the three
families on a synthetic deterministic target task (Bayes risk ``0`` for
deterministic labels and cross entropy) and compares the minimum achievable
loss, i.e. the empirical approximation error ordering of Theorem 4.2 /
Proposition 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_generator import MaskNet

__all__ = [
    "encode_shared_mask",
    "verify_shared_mask_inclusion",
    "verify_sample_specific_inclusion",
    "empirical_approximation_error_experiment",
]


def encode_shared_mask(
    mask: torch.Tensor,
    mask_net: MaskNet,
    snap_to_patches: bool = True,
) -> MaskNet:
    """Construct the ``F_shr`` element inside ``F_smm`` (proof of Prop. 4.3).

    Every convolution weight is set to zero and the last-layer affine term is
    set to the shared binary mask, so that ``f_mask(r(x)) = M`` for all ``x``.

    Two formalisations are supported, exactly as in Appendix B.2 where the last
    layer is written ``W_last f''(r(x)) + b_last`` with ``b_last`` of the
    output size:

    * ``mask_net.generator.spatial_bias`` is ``True``: the generator carries an
      explicit output-space bias, so *any* shared mask (including the
      spatially varying Narrow / Medium / Pad borders) is representable.
    * otherwise: only masks that are constant per channel are representable
      (``M = c * J``); this covers the watermarking / "Full" mask of Bahng et
      al. (2022), i.e. the main shared-mask baseline of the paper.

    With ``l > 0`` pooling layers the generated mask is constant inside every
    ``2**l x 2**l`` patch (Section 3.3), so the shared masks that are exactly
    representable are those equal to their patch-constant version;
    ``snap_to_patches`` replaces ``M`` by that version (a no-op when ``M`` is
    already patch aligned, and exactly ``M`` when ``l = 0``).
    """
    mask = mask.detach().float()
    if mask.dim() == 4:
        mask = mask[0, 0]
    if mask.shape != torch.Size(mask_net.image_size):
        mask = F.interpolate(
            mask[None, None], size=mask_net.image_size, mode="nearest"
        )[0, 0]

    patch = mask_net.patch_size
    if patch > 1:
        h, w = mask.shape
        padded = F.pad(
            mask[None, None],
            (0, (patch - w % patch) % patch, 0, (patch - h % patch) % patch),
            mode="replicate",
        )
        low_res = F.max_pool2d(padded, kernel_size=patch, stride=patch)[0, 0]
        snapped = low_res.repeat_interleave(patch, 0).repeat_interleave(patch, 1)[:h, :w]
        if snap_to_patches:
            mask = snapped
        elif not torch.equal(snapped, mask):  # pragma: no cover - defensive
            raise ValueError(
                "the shared mask is not constant inside every patch and cannot be "
                "represented exactly when pooling layers are used: use l=0 or "
                "snap_to_patches=True"
            )
    else:
        low_res = mask

    with torch.no_grad():
        for p in mask_net.parameters():
            p.zero_()
        generator = mask_net.generator
        if generator.output_bias is not None:
            generator.output_bias.copy_(
                low_res.reshape(1, 1, *low_res.shape[-2:]).expand(1, generator.out_channels,
                                                                 *low_res.shape[-2:])
            )
        else:
            _set_channel_constant_bias(mask_net, low_res)
    return mask_net


def _set_channel_constant_bias(mask_net: MaskNet, low_res_mask: torch.Tensor) -> MaskNet:
    """Set the last-layer bias for masks that are constant per channel."""
    per_channel = low_res_mask.reshape(-1)
    if per_channel.numel() > 1 and not torch.allclose(
        per_channel, per_channel[0].expand_as(per_channel)
    ):
        raise ValueError(
            "this mask is spatially varying and cannot be encoded by a per-channel "
            "convolution bias; build the mask generator with spatial_bias=True to "
            "follow the output-space formalisation of Appendix B.2"
        )
    last_conv = mask_net.generator.convs[-1]
    last_conv.bias.copy_(per_channel[0].expand(last_conv.out_channels))
    return mask_net


def snap_mask_to_patches(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patch-constant version of ``mask`` (the exactly representable masks)."""
    mask = mask.float()
    if mask.dim() == 4:
        mask = mask[0, 0]
    if patch_size <= 1:
        return mask
    h, w = mask.shape
    padded = F.pad(
        mask[None, None],
        (0, (patch_size - w % patch_size) % patch_size,
         0, (patch_size - h % patch_size) % patch_size),
        mode="replicate",
    )
    low = F.max_pool2d(padded, kernel_size=patch_size, stride=patch_size)[0, 0]
    return low.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)[:h, :w]


@torch.no_grad()
def verify_shared_mask_inclusion(
    image_size: Tuple[int, int] = (32, 32),
    mask: Optional[torch.Tensor] = None,
    num_pool_layers: int = 2,
    batch_size: int = 4,
    seed: int = 0,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """End-to-end numerical check of Proposition 4.3 (``F_shr subseteq F_smm``).

    A shared binary mask ``M`` is encoded into the mask generator of an SMM
    module (all weights zero, last-layer bias = ``M``); both ``f_in`` modules
    must then produce the identical reprogrammed input for arbitrary samples
    and for the same shared pattern ``delta``.
    """
    from .reprogram import InputReprogramming, InputReprogrammingConfig

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, *image_size, generator=g)
    if mask is None:
        mask = torch.zeros(*image_size)
        mask[:4, :] = 1.0  # an arbitrary binary shared mask

    common = dict(image_size=image_size, num_layers=5, hidden_channels=(8, 8, 8))
    smm = InputReprogramming(InputReprogrammingConfig(
        variant="smm", num_pool_layers=num_pool_layers, spatial_bias=True, **common))
    shared = InputReprogramming(InputReprogrammingConfig(
        variant="shared", mask_kind="full", num_pool_layers=num_pool_layers, **common))

    patch = 2 ** num_pool_layers
    representable = snap_mask_to_patches(mask, patch)
    shared.shared_mask.copy_(representable.reshape(1, 1, *image_size))
    encode_shared_mask(representable, smm.mask_net)

    with torch.no_grad():
        delta = torch.randn(1, 3, *image_size, generator=g)
        shared.delta.copy_(delta)
        smm.delta.copy_(delta)
        out_shared = shared(x)
        out_smm = smm(x)
        generated = smm.make_mask(x)

    diff = (out_shared - out_smm).abs().max().item()
    mask_diff = (generated - representable.reshape(1, 1, *image_size)).abs().max().item()
    return {
        "max_abs_diff": diff,
        "mask_abs_diff": mask_diff,
        "patch_size": patch,
        "included": float(diff <= tol and mask_diff <= tol),
    }


@torch.no_grad()
def verify_watermark_inclusion(
    image_size: Tuple[int, int] = (32, 32),
    num_pool_layers: int = 3,
    batch_size: int = 4,
    seed: int = 0,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Special case of Proposition 4.3 with the plain CNN (no spatial bias).

    The full watermarking mask ``M = J`` is constant per channel, hence exactly
    representable by the last-layer bias of a standard convolutional generator:
    ``F_watermark subseteq F_smm``.  This is the shared-mask baseline
    ("Full" / "Only delta") of Table 1 and Table 3.
    """
    from .reprogram import InputReprogramming, InputReprogrammingConfig

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, *image_size, generator=g)
    common = dict(image_size=image_size, num_layers=5, hidden_channels=(8, 8, 8))
    smm = InputReprogramming(InputReprogrammingConfig(
        variant="smm", num_pool_layers=num_pool_layers, **common))
    full = InputReprogramming(InputReprogrammingConfig(
        variant="shared", mask_kind="full", num_pool_layers=num_pool_layers, **common))

    encode_shared_mask(torch.ones(*image_size), smm.mask_net)
    with torch.no_grad():
        delta = torch.randn(1, 3, *image_size, generator=g)
        full.delta.copy_(delta)
        smm.delta.copy_(delta)
        diff = (full(x) - smm(x)).abs().max().item()
        mask_err = (smm.make_mask(x) - 1.0).abs().max().item()
    return {"max_abs_diff": diff, "mask_error": mask_err, "included": float(diff <= tol)}


@torch.no_grad()
def verify_sample_specific_inclusion(
    image_size: Tuple[int, int] = (32, 32),
    batch_size: int = 4,
    num_pool_layers: int = 2,
    seed: int = 0,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """``F_sp subseteq F_smm``: setting ``delta = J`` recovers ``f_mask`` alone."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 3, *image_size, generator=g)
    mask_net = MaskNet(
        image_size=image_size, num_pool_layers=num_pool_layers, hidden_channels=(8, 8, 8)
    )
    delta = torch.ones(1, 3, *image_size)
    sp = x + mask_net(x)
    smm = x + delta * mask_net(x)
    diff = (sp - smm).abs().max().item()
    return {"max_abs_diff": diff, "included": float(diff <= tol)}


# --------------------------------------------------------------------------- #
# Empirical approximation error (Definition 4.1 / Theorem 4.2)
# --------------------------------------------------------------------------- #
class _FrozenPretrainedModel(nn.Module):
    """A small frozen 'pre-trained' classifier over a 1-channel image input."""

    def __init__(self, image_size: int, num_classes: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(image_size * image_size // 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else p.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x).flatten(1)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)


@dataclass
class _Family:
    name: str
    delta: Optional[nn.Parameter]
    mask_net: Optional[nn.Module]
    fixed_mask: Optional[torch.Tensor]

    def parameters(self) -> List[nn.Parameter]:
        params = []
        if self.delta is not None:
            params.append(self.delta)
        if self.mask_net is not None:
            params.extend(self.mask_net.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_net is not None:
            m = self.mask_net(x)
        else:
            m = self.fixed_mask
        if self.delta is None:
            return x + m
        return x + self.delta * m


def empirical_approximation_error_experiment(
    num_samples: int = 2048,
    image_size: int = 8,
    num_classes: int = 10,
    steps: int = 400,
    restarts: int = 3,
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, object]:
    """Compare the best achievable loss of ``F_shr``, ``F_sp`` and ``F_smm``.

    The target task is deterministic (``y = g(x)``), hence the Bayes risk is
    zero and the minimum cross entropy of a family is exactly its approximation
    error.  Labels are the class of the quadrant/sign pattern of a random
    projection of the image, which is only partially expressible by a *single*
    shared mask -- the intuition behind Figure 1/2 of the paper.
    """
    device = torch.device(device)
    torch.manual_seed(seed)

    x = torch.randn(num_samples, 1, image_size, image_size)
    # Deterministic, sample-dependent target function.
    w = torch.randn(1, image_size, image_size)
    s1 = (x * w).sum(dim=(1, 2, 3))
    s2 = (x * w.flip(-1)).sum(dim=(1, 2, 3))
    y = (
        (s1 > 0).long() * (num_classes // 2)
        + (s2 > 0).long() * (num_classes // 4)
        + ((s1.abs() > 1).long())
    ).clamp(0, num_classes - 1)
    y = y % num_classes

    pretrained = _FrozenPretrainedModel(image_size, num_classes, seed=seed).to(device)
    for p in pretrained.parameters():
        p.requires_grad_(False)
    x = x.to(device)
    y = y.to(device)

    mask_shape = (1, 1, image_size, image_size)
    border = torch.zeros(mask_shape, device=device)
    border[..., : image_size // 4, :] = 1.0
    border[..., -image_size // 4 :, :] = 1.0

    best: Dict[str, float] = {}
    for restart in range(restarts):
        torch.manual_seed(seed + restart)
        families = {
            "shr": _Family("shr", nn.Parameter(torch.zeros(mask_shape, device=device)), None,
                           border.to(device)),
            "sp": _Family("sp", None,
                          MaskNet(image_size=(image_size, image_size), in_channels=1,
                                  out_channels=1, num_pool_layers=2,
                                  hidden_channels=(8, 16, 16)).to(device),
                          None),
            "smm": _Family("smm", nn.Parameter(torch.zeros(mask_shape, device=device)),
                           MaskNet(image_size=(image_size, image_size), in_channels=1,
                                   out_channels=1, num_pool_layers=2,
                                   hidden_channels=(8, 16, 16)).to(device),
                           None),
        }
        for name, family in families.items():
            optimizer = torch.optim.Adam(family.parameters(), lr=lr)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = pretrained(family.forward(x))
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final = float(F.cross_entropy(pretrained(family.forward(x)), y).item())
            best[name] = min(best.get(name, float("inf")), final)
            if verbose:
                print(f"restart {restart} family {name}: loss={final:.4f} (best={best[name]:.4f})")

    return {
        "approximation_error": best,
        "ordering_holds": best["smm"] <= best["shr"] + 1e-8 and best["smm"] <= best["sp"] + 1e-8,
        "bayes_risk": 0.0,
        "num_samples": num_samples,
    }
