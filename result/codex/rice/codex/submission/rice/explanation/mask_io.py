"""Checkpoint helpers shared by the explanation methods."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import torch

from rice.explanation.mask_trainer import MaskActorCritic
from rice.networks import MaskNet


def save_mask_net(mask_net: torch.nn.Module, path: str, meta: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": mask_net.state_dict(),
            "obs_dim": getattr(mask_net, "obs_dim", None),
            "hidden": infer_hidden(mask_net.state_dict()),
            "meta": meta or {},
        },
        path,
    )


def infer_hidden(state_dict) -> tuple:
    """Recover the hidden sizes of a mask network from its state dict."""
    layers = []
    for key in state_dict:
        match = re.match(r"^net\.(\d+)\.weight$", key)
        if match:
            layers.append((int(match.group(1)), int(state_dict[key].shape[0])))
    if not layers:
        return ()
    layers.sort()
    sizes = [size for _, size in layers]
    # the last entry is the 2-action output head
    return tuple(sizes[:-1])


def load_mask_net(
    path: str, hidden=(64, 64), with_value_head: bool = True, map_location: str = "cpu"
):
    """Load a mask network.

    Works both for checkpoints produced by this repository and for the official
    StateMask checkpoints (``with_value_head=False``), which only contain the
    mask policy.
    """
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt.get("state_dict", ckpt)
    obs_dim = ckpt.get("obs_dim")
    if obs_dim is None:
        first = next(iter(state_dict.values()))
        obs_dim = first.shape[-1]
    ckpt_hidden = ckpt.get("hidden") if isinstance(ckpt, dict) else None
    if not ckpt_hidden:
        ckpt_hidden = infer_hidden(state_dict)
    if ckpt_hidden:
        hidden = tuple(int(h) for h in ckpt_hidden)
    cls = MaskActorCritic if with_value_head else MaskNet
    model = cls(obs_dim, hidden=hidden)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) and not with_value_head:
        pass
    model.eval()
    return model


def load_mask_meta(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Metadata stored next to a mask checkpoint (samples, training time, ...)."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        return dict(ckpt.get("meta", {}) or {})
    return {}
