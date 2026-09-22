"""Exponential moving average (EMA) of model parameters.

The paper does not mention EMA for the reported FID numbers (Appendix B / Addendum
say nothing about weight averaging), so EMA is *optional* engineering glue: it is
provided here because :mod:`si.utils` advertises ``EMA``/``update_ema``/``copy_params``
and because weight averaging is a convenient, cheap way to stabilise FID when a run
is stopped early.

Conventions
-----------
* EMA tracks a *shadow* copy of ``state_dict`` entries (both parameters and buffers,
  with buffers optionally copied verbatim).
* ``decay`` is the per-update coefficient: ``theta_ema <- decay * theta_ema +
  (1 - decay) * theta``.
* A warmup-stabilised decay (``min(decay, (1 + step) / (10 + step))``) is available and
  is the standard trick that avoids a bias towards the random initialisation during the
  first few hundred steps.
* Everything works on plain ``torch.nn.Module``s, on ``state_dict``s, and on devices
  via ``device``/``dtype`` casting of the shadow tensors (useful to keep the shadow copy
  in fp32 while the model runs in lower precision).

Nothing here is on the critical path of Algorithm 1; the helpers degrade gracefully and
are safe to ignore.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["EMA", "update_ema", "copy_params"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip common wrappers (DDP/DDP-like ``.module`` and ``torch.compile``)."""
    if model is None:
        return model
    for attr in ("module", "_orig_mod"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            return _unwrap_model(inner)
    return model


def _state_items(model: Union[nn.Module, Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if isinstance(model, nn.Module):
        return dict(_unwrap_model(model).state_dict())
    if isinstance(model, Mapping):
        return dict(model)
    raise TypeError(f"Expected nn.Module or state_dict-like mapping, got {type(model)!r}")


def copy_params(
    target: Union[nn.Module, Mapping[str, torch.Tensor]],
    source: Union[nn.Module, Mapping[str, torch.Tensor]],
    *,
    copy_buffers: bool = True,
    model: Optional[nn.Module] = None,
) -> Union[nn.Module, Dict[str, torch.Tensor]]:
    """Copy parameter values from ``source`` into ``target`` in place.

    Parameters
    ----------
    target:
        ``nn.Module`` (mutated in place) or a mapping of tensors (mutated in place).
    source:
        ``nn.Module`` or mapping providing the values.
    copy_buffers:
        When ``True`` (default) non-parameter buffers are copied as well.
    model:
        Optional alias for ``target`` when ``target`` is a mapping (kept for API
        symmetry with :class:`EMA`).

    Returns
    -------
    The ``target`` object (module or mapping) for convenience.
    """
    if target is None:
        target = model
    if target is None:
        raise ValueError("copy_params requires a target module/state_dict")

    src = _state_items(source)

    if isinstance(target, nn.Module):
        tgt_sd = _unwrap_model(target).state_dict()
        with torch.no_grad():
            for name, tensor in tgt_sd.items():
                if name not in src:
                    continue
                if not copy_buffers and not _is_parameter(_unwrap_model(target), name):
                    continue
                tensor.copy_(src[name].to(tensor.device, tensor.dtype))
        return target

    if isinstance(target, Mapping):
        with torch.no_grad():
            for name, tensor in target.items():
                if name not in src:
                    continue
                tensor.copy_(src[name].to(tensor.device, tensor.dtype))
        return target

    raise TypeError(f"Expected nn.Module or mapping target, got {type(target)!r}")


def _is_parameter(model: nn.Module, name: str) -> bool:
    for pname, _ in model.named_parameters():
        if pname == name:
            return True
    return False


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of a module's parameters.

    Example
    -------
    >>> ema = EMA(model, decay=0.9999)
    >>> for step, batch in enumerate(loader):
    ...     ...                      # train `model`
    ...     ema.update(model)

    Parameters
    ----------
    model:
        Model whose parameters (and optionally buffers) are averaged.
    decay:
        Target per-update decay in ``[0, 1)``. Larger = slower.
    warmup:
        Apply the ``min(decay, (1 + step) / (10 + step))`` stabilised decay for the
        first steps (default ``True``).
    update_buffers:
        Also average buffers whose dtype is floating point (default ``False``:
        buffers are *copied*, e.g. GroupNorm has none, BN would need care).
    device / dtype:
        Optional cast of the shadow copy (e.g. keep fp32 shadow for a bf16 model).
    copy_buffers:
        At construction time, copy buffers into the shadow (informational).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        *,
        warmup: bool = True,
        update_buffers: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        copy_buffers: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"EMA expects an nn.Module, got {type(model)!r}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.update_buffers = bool(update_buffers)
        self.step = 0
        self.model_ref: Optional[nn.Module] = None  # no strong reference (avoid loops)

        base = _unwrap_model(model)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._is_param: Dict[str, bool] = {}
        param_names = {n for n, _ in base.named_parameters()}

        for name, tensor in base.state_dict().items():
            keep = name in param_names or (self.update_buffers and tensor.is_floating_point())
            if not keep:
                continue
            t = tensor.detach().clone()
            if device is not None:
                t = t.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            self.shadow[name] = t
            self._is_param[name] = name in param_names

        if copy_buffers and not self.update_buffers:
            # remembered only for verbose repr; nothing else to do
            pass

    # -- core -------------------------------------------------------------
    @property
    def effective_decay(self) -> float:
        """Decay actually used at the current step (warmup aware)."""
        if not self.warmup:
            return self.decay
        return float(min(self.decay, (1.0 + self.step) / (10.0 + self.step)))

    @torch.no_grad()
    def update(
        self,
        model: Optional[nn.Module] = None,
        *,
        decay: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        """Update the shadow copy from ``model`` (or the constructor model)."""
        if model is None:
            model = self.model_ref
        if model is None:
            raise ValueError("EMA.update requires a model (none supplied at construction)")
        if step is not None:
            self.step = int(step)
        d = float(self.effective_decay if decay is None else decay)

        src = _state_items(model)
        for name, shadow in self.shadow.items():
            value = src.get(name, None)
            if value is None:
                continue
            value = value.detach().to(shadow.device, shadow.dtype)
            shadow.mul_(d).add_(value, alpha=1.0 - d)

        self.step += 1

    # alias used by some codebases
    def step_forward(self, model: Optional[nn.Module] = None) -> None:  # pragma: no cover
        self.update(model)

    # -- application ------------------------------------------------------
    @torch.no_grad()
    def copy_to(
        self,
        model: nn.Module,
        *,
        copy_buffers: bool = True,
    ) -> nn.Module:
        """Write the averaged weights into ``model`` (in place) and return it."""
        sd = dict(_unwrap_model(model).state_dict())
        for name, shadow in self.shadow.items():
            if name not in sd:
                continue
            if not copy_buffers and not self._is_param.get(name, False):
                continue
            sd[name].copy_(shadow.to(sd[name].device, sd[name].dtype))
        return model

    # convenience aliases
    copy_params_to = copy_to

    def store(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Backup the *live* state dict of ``model`` (so it can be restored)."""
        return {k: v.detach().clone() for k, v in _state_items(model).items()}

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Mapping[str, torch.Tensor]) -> nn.Module:
        """Restore a backup produced by :meth:`store` into ``model``."""
        sd = dict(_unwrap_model(model).state_dict())
        for name, value in backup.items():
            if name in sd:
                sd[name].copy_(value.to(sd[name].device, sd[name].dtype))
        return model

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable state (shadow tensors kept on CPU for portability)."""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "update_buffers": self.update_buffers,
            "step": self.step,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
            "is_param": dict(self._is_param),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "EMA":
        self.decay = float(state.get("decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))
        self.update_buffers = bool(state.get("update_buffers", self.update_buffers))
        self.step = int(state.get("step", self.step))
        shadow = state.get("shadow", {})
        self.shadow = {k: v.detach().clone() for k, v in shadow.items()}
        self._is_param = dict(state.get("is_param", {k: True for k in self.shadow}))
        return self

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, warmup={self.warmup}, step={self.step}, "
            f"tensors={len(self.shadow)})"
        )


def update_ema(
    ema: Union[EMA, Mapping[str, torch.Tensor], nn.Module],
    model: Optional[nn.Module] = None,
    decay: Optional[float] = None,
) -> None:
    """Functional wrapper around :meth:`EMA.update`.

    Accepts either an :class:`EMA` instance (``update_ema(ema, model)``) or a raw shadow
    mapping (``update_ema(shadow, model, decay=0.999)``).
    """
    if isinstance(ema, EMA):
        ema.update(model, decay=decay)
        return
    if model is None:
        raise ValueError("update_ema(shadow, model) requires a source model")
    if decay is None:
        raise ValueError("update_ema requires an explicit decay when given a mapping")
    shadow = ema  # type: ignore[assignment]
    src = _state_items(model)
    with torch.no_grad():
        for name, tensor in shadow.items():  # type: ignore[union-attr]
            value = src.get(name, None)
            if value is None:
                continue
            tensor.mul_(decay).add_(value.detach().to(tensor.device, tensor.dtype), alpha=1 - decay)


def ema_from_config(model: nn.Module, config: Optional[Mapping[str, Any]] = None) -> Optional[EMA]:
    """Build an :class:`EMA` from a config mapping, or ``None`` when disabled.

    Recognised keys: ``enabled``/``use_ema``, ``decay``, ``warmup``, ``update_buffers``,
    ``device``, ``dtype``.
    """
    config = dict(config or {})
    enabled = bool(config.get("enabled", config.get("use_ema", False)))
    if not enabled:
        return None
    dtype = config.get("dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    return EMA(
        model,
        decay=float(config.get("decay", 0.9999)),
        warmup=bool(config.get("warmup", True)),
        update_buffers=bool(config.get("update_buffers", False)),
        device=config.get("device", None),
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    ema = EMA(model, decay=0.9, warmup=False)

    # at construction the shadow equals the weights
    for name, value in model.state_dict().items():
        if name in ema.shadow:
            assert torch.allclose(value, ema.shadow[name]), name

    # move the model, EMA should lag behind
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = ema.shadow["0.weight"].clone()
    ema.update(model)
    delta = (ema.shadow["0.weight"] - before)
    assert torch.all(delta >= 0), "EMA moved in the wrong direction"
    assert torch.allclose(delta, torch.full_like(delta, 0.1), atol=1e-6), delta.mean()

    # convergence: many updates -> shadow ~= live params
    small = nn.Linear(3, 2)
    ema2 = EMA(small, decay=0.5)
    for _ in range(60):
        ema2.update(small)
    for name, value in small.state_dict().items():
        assert torch.allclose(value, ema2.shadow[name], atol=1e-6), name

    # copy_to writes averaged weights back
    ema2.copy_to(small)

    # state dict round-trip
    sd = ema2.state_dict()
    ema3 = EMA(small, decay=0.1)
    ema3.load_state_dict(sd)
    assert ema3.step == ema2.step and ema3.decay == ema2.decay

    # functional mapping form
    shadow = {k: v.clone() for k, v in small.state_dict().items()}
    update_ema(shadow, small, decay=0.5)
    assert torch.allclose(shadow["weight"], small.weight, atol=1e-6)

    # wrapped model forms (DDP-like)
    class Wrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.module = inner

    wrapped = Wrapper(small)
    ema4 = EMA(wrapped, decay=0.9, warmup=False)
    ema4.update(wrapped)
    copy_params(small, ema4.shadow)
    assert ema_from_config(small, {}) is None
    assert isinstance(ema_from_config(small, {"enabled": True}), EMA)

    print("si.utils.ema self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
