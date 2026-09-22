"""Velocity / score models used as b_hat_t(x, xi) and g_hat_t(x, xi).

``VelocityModel`` turns an image-to-image backbone (the U-Net of Appendix B,
or any other ``nn.Module`` with the signature ``forward(x, time, labels)``)
into the conditional velocity b_hat_t(x_t, xi) of the paper:

* the conditioning signal xi (low-resolution image for super-resolution, the
  missingness mask for in-painting, nothing for the independent baseline) is
  appended to the channels of x_t at *every* call, as in Appendix B;
* the class label is passed to the class embedding of the U-Net;
* for in-painting the output is multiplied by the mask 1 - xi so that the
  observed pixels are never modified, which enforces the structural property
  b_t(x, xi) = 0 on the observed pixels discussed in Section 4.1.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class VelocityModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        data_channels: int = 3,
        cond_channels: int = 0,
        masks_output: bool = False,
        time_scale: float = 1000.0,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.data_channels = data_channels
        self.cond_channels = cond_channels
        self.masks_output = masks_output
        self.time_scale = float(time_scale)
        self.guidance_scale = float(guidance_scale)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("this velocity model expects conditioning channels (xi)")
            x = torch.cat([x, cond], dim=1)
        time = (t * self.time_scale) if t.dim() == 1 else (t * self.time_scale).reshape(t.shape[0])
        v = self.backbone(x, time, labels)
        if self.masks_output and mask is not None:
            v = v * mask
        return v

    # ------------------------------------------------------------------
    @torch.no_grad()
    def guided_velocity(
        self,
        x: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        guidance_scale: Optional[float] = None,
    ) -> Tensor:
        """Classifier-free guided velocity (useful at sampling time).

        v = v_uncond + w * (v_cond - v_uncond), with the null class index
        ``backbone.null_class``.  ``w = 1`` reduces to the plain model.
        """
        w = self.guidance_scale if guidance_scale is None else guidance_scale
        v = self.forward(x, t, cond, labels, mask)
        if w == 1.0 or labels is None or getattr(self.backbone, "num_classes", None) is None:
            return v
        null = torch.full_like(labels, self.backbone.null_class)
        v_null = self.forward(x, t, cond, null, mask)
        return v_null + w * (v - v_null)


def build_velocity_model(cfg: dict, coupling_info: dict) -> VelocityModel:
    """Build the velocity model described by an experiment configuration."""
    from .unet import Unet, unet_imagenet

    model_cfg = dict(cfg.get("model", {}))
    kind = model_cfg.pop("kind", "unet_imagenet")
    data_channels = int(model_cfg.pop("data_channels", cfg.get("data", {}).get("channels", 3)))
    time_scale = float(model_cfg.pop("time_scale", 1000.0))
    cond_channels = int(coupling_info.get("cond_channels", 0))
    num_classes = model_cfg.pop("num_classes", cfg.get("data", {}).get("num_classes", 1000))
    image_size = int(model_cfg.pop("image_size", cfg.get("data", {}).get("image_size", 256)))

    backbone_channels = data_channels + cond_channels
    if kind == "unet_imagenet":
        backbone = unet_imagenet(
            num_classes=num_classes,
            channels=backbone_channels,
            out_dim=data_channels,
            image_size=image_size,
            **model_cfg,
        )
    elif kind == "unet":
        backbone = Unet(
            channels=backbone_channels,
            out_dim=data_channels,
            num_classes=num_classes,
            image_size=image_size,
            **model_cfg,
        )
    else:
        raise ValueError(f"unknown model kind {kind!r}")

    return VelocityModel(
        backbone,
        data_channels=data_channels,
        cond_channels=cond_channels,
        masks_output=bool(coupling_info.get("masks_output", False)),
        time_scale=time_scale,
        guidance_scale=float(cfg.get("sampling", {}).get("guidance_scale", 1.0)),
    )


__all__ = ["VelocityModel", "build_velocity_model"]
