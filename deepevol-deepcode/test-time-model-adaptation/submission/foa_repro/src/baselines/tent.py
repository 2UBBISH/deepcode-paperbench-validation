"""TENT: Fully Test-Time Adaptation by Entropy Minimization (Wang et al., 2021).

Thin, self-contained adapter that reproduces the *exact* configuration the FOA
paper (Appendix B.2) uses for TENT:

    "We follow all hyperparameters that are set in Tent unless it does not
     provide. Specifically, we use SGD as the update rule, with a momentum of
     0.9, batch size of 64 and the learning rate of 0.001. The trainable
     parameters are all affine parameters of layer normalization layers."

Reference implementation: https://github.com/DequanWang/tent (DequanWang/tent).

Adaptation is standard online TENT: for every test batch
  1. forward the batch (using the *current* parameters) to obtain logits,
  2. compute the mean softmax entropy,
  3. backpropagate and take one SGD step on the layer-norm affine parameters,
  4. predict with the updated model.
No pseudo-labels and no source data are used.

The wrapper is deliberately tolerant about the model object it receives: it
accepts a plain ``timm``/``torch.nn`` model or the FOA ``ViTWithCLSFeatures``
wrapper (whose real modules live under ``.model``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "TENT",
    "select_norm_affine_params",
    "entropy_loss",
    "softmax_entropy",
    "TENTConfig",
]

# Appendix B.2 defaults -------------------------------------------------------
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPISODIC = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unwrap(model: Any) -> nn.Module:
    """Return the underlying ``nn.Module`` of a (possibly wrapped) model."""
    if isinstance(model, nn.Module):
        # FOA wrapper stores the actual timm model under ``.model``.
        inner = getattr(model, "model", None)
        if isinstance(inner, nn.Module) and inner is not model:
            return inner
        return model
    raise TypeError(f"TENT expects an nn.Module, got {type(model)!r}")


def _norm_affine_names(module: nn.Module) -> List[str]:
    """Names of the affine parameters (weight/bias) of every LayerNorm."""
    names: List[str] = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.LayerNorm):
            for pname, param in sub.named_parameters(recurse=False):
                if param.requires_grad is not None:  # always true
                    names.append(f"{name}.{pname}" if name else pname)
    return names


def select_norm_affine_params(model: Any) -> List[nn.Parameter]:
    """Collect all LayerNorm affine parameters (TENT's trainable set).

    Works for a plain timm ViT and for the FOA ``ViTWithCLSFeatures`` wrapper.
    """
    module = _unwrap(model)
    params: List[nn.Parameter] = []
    for _, sub in module.named_modules():
        if isinstance(sub, nn.LayerNorm):
            for p in sub.parameters(recurse=False):
                params.append(p)
    return params


def softmax_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy of each sample's softmax distribution -> ``[B]``."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(eps))
    return -(probs * log_probs).sum(dim=-1)


def entropy_loss(logits: torch.Tensor, eps: float = 1e-12, reduction: str = "mean") -> torch.Tensor:
    """TENT's objective: mean softmax entropy over the batch."""
    ent = softmax_entropy(logits, eps=eps)
    if reduction == "mean":
        return ent.mean()
    if reduction == "sum":
        return ent.sum()
    return ent


# ---------------------------------------------------------------------------
# configuration container
# ---------------------------------------------------------------------------
class TENTConfig:
    """Hyper-parameters of the TENT baseline (Appendix B.2)."""

    def __init__(
        self,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        episodic: bool = DEFAULT_EPISODIC,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        entropy_eps: float = 1e-12,
        adapt_after_loss: bool = True,
    ) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.batch_size = int(batch_size)
        self.episodic = bool(episodic)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.entropy_eps = float(entropy_eps)
        self.adapt_after_loss = bool(adapt_after_loss)

    @classmethod
    def from_config(cls, cfg: Any) -> "TENTConfig":
        """Build from an FOA config dict (``cfg.baselines.tent`` if present)."""
        bl = None
        if cfg is not None:
            try:
                bl = cfg["baselines"]["tent"]  # type: ignore[index]
            except Exception:
                bl = getattr(getattr(cfg, "baselines", None), "tent", None)
        get = _getter(bl)
        return cls(
            lr=get("lr", DEFAULT_LR),
            momentum=get("momentum", DEFAULT_MOMENTUM),
            batch_size=get("batch_size", DEFAULT_BATCH_SIZE),
            episodic=get("episodic", DEFAULT_EPISODIC),
            weight_decay=get("weight_decay", 0.0),
            nesterov=get("nesterov", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _getter(obj: Any):
    def get(key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        try:
            val = obj[key]  # type: ignore[index]
        except Exception:
            val = getattr(obj, key, default)
        return default if val is None else val

    return get


# ---------------------------------------------------------------------------
# main wrapper
# ---------------------------------------------------------------------------
class TENT:
    """Online entropy-minimization test-time adaptation on norm affine params.

    Typical use inside an online TTA stream::

        tent = TENT(model, cfg)
        for images, targets in stream:
            logits = tent.step(images)          # adapt once, then predict
            acc.update(logits, targets)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[TENTConfig] = None,
        *,
        lr: float = DEFAULT_LR,
        momentum: float = DEFAULT_MOMENTUM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[Any] = None,
        episodic: bool = DEFAULT_EPISODIC,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.wrapped = model
        self.module = _unwrap(model)
        self.config = config or TENTConfig(
            lr=lr,
            momentum=momentum,
            batch_size=batch_size,
            episodic=episodic,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.lr = self.config.lr
        self.momentum = self.config.momentum
        self.batch_size = self.config.batch_size
        self.episodic = self.config.episodic
        self.grad_clip = grad_clip

        if device is None:
            try:
                device = next(self.module.parameters()).device
            except StopIteration:  # pragma: no cover - degenerate model
                device = torch.device("cpu")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # TENT trains ONLY the LayerNorm affine parameters.
        self.params: List[nn.Parameter] = select_norm_affine_params(self.module)
        if not self.params:
            raise RuntimeError(
                "TENT found no LayerNorm affine parameters to adapt; the model "
                "does not look like a ViT/transformer backbone."
            )
        self.optimizer = torch.optim.SGD(
            self.params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.nesterov,
        )
        self._init_state = self._snapshot_params()
        self._num_steps = 0
        self.module.eval()

    # -- state management ---------------------------------------------------
    def _snapshot_params(self) -> List[torch.Tensor]:
        return [p.detach().clone() for p in self.params]

    def reset(self) -> None:
        """Restore the source parameters and clear optimizer/momentum state."""
        with torch.no_grad():
            for p, s in zip(self.params, self._init_state):
                p.copy_(s)
        self.optimizer = torch.optim.SGD(
            self.params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=self.config.nesterov,
        )
        self._num_steps = 0

    def load_source_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load source weights and refresh the restore snapshot."""
        missing, unexpected = self.module.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("TENT load_source_state missing keys: %d", len(missing))
        if unexpected:
            logger.debug("TENT load_source_state unexpected keys: %d", len(unexpected))
        self._init_state = self._snapshot_params()
        self.reset()

    # -- forward helpers ----------------------------------------------------
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward through the model and return logits (grad-enabled)."""
        if hasattr(self.wrapped, "forward") and self.wrapped is not self.module:
            out = self.wrapped(images)  # FOA wrapper -> logits
            if isinstance(out, dict):
                out = out.get("logits", out.get("out"))
            return out
        return self.module(images)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Inference only (no adaptation step)."""
        images = images.to(self.device, non_blocking=True)
        out = self._forward(images)
        if isinstance(out, dict):
            out = out.get("logits", out)
        return out

    def loss(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(entropy_loss, logits)`` for one batch (grad enabled)."""
        logits = self._forward(images)
        return entropy_loss(logits, eps=self.config.entropy_eps), logits

    def adapt(self, images: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """One TENT update on ``images``; returns ``(detached logits, loss)``."""
        self.module.train() if False else None  # ViT has no dropout by default
        logits = self._forward(images)
        loss = entropy_loss(logits, eps=self.config.entropy_eps)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self._num_steps += 1
        return logits.detach(), float(loss.detach().cpu())

    def step(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """TENT's per-batch routine: adapt then predict with updated params.

        Note the standard TENT protocol predicts with the *updated* parameters.
        Passing ``targets`` only enables optional accuracy bookkeeping; the
        adaptation itself is label-free.
        """
        images = images.to(self.device, non_blocking=True)
        self.adapt(images)
        with torch.no_grad():
            logits = self._forward(images)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits)
        return logits

    __call__ = step

    # -- convenience --------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return self._num_steps

    def trainable_parameter_names(self) -> List[str]:
        module = self.module
        trainable = {id(p) for p in self.params}
        return [n for n, p in module.named_parameters() if id(p) in trainable]

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "num_steps": self._num_steps,
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self._num_steps = int(state.get("num_steps", 0))


# ---------------------------------------------------------------------------
# factory used by the runner scripts
# ---------------------------------------------------------------------------
def build_tent(model: Any = None, cfg: Any = None, **kwargs) -> TENT:
    """Config-driven factory: ``build_tent(model, cfg)``.

    Accepts either a model or an FOA config (or both); hyper-parameters are
    read from ``cfg.baselines.tent`` when available and can always be
    overridden with explicit keyword arguments.
    """
    if model is None and isinstance(cfg, torch.nn.Module):
        model = cfg
        cfg = None
    if model is None:
        raise ValueError("build_tent requires a model instance")
    config = TENTConfig.from_config(cfg)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return TENT(model, config)


def build(model: Any = None, cfg: Any = None, **kwargs) -> TENT:  # noqa: A001
    return build_tent(model, cfg, **kwargs)


build_baseline = build_tent
