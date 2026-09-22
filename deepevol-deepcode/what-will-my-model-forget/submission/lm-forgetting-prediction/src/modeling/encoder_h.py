"""Encoding function ``h`` used by both trainable forecasting models.

Paper specification
-------------------
Sec. 3.2 ("Trainable Logit-Based Forecasting Model")::

    ... we examine a significantly simplified alternative of Eqn. 2 by substituting
    Theta(x_j, x_i) Theta^{-1}(x_i, x_i) with a trainable kernel
    Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T, where h: x, y -> R^{T x d}
    is an encoding function that maps concatenation of inputs and outputs x, y to a
    low-dimensional vector in R^d, where T is the length of the output y.  We
    implement h with a trainable LM and extract its representation of output tokens
    in the final layer as h(x, y).

Sec. 3.3 ("Representation-Based Forecasting")::

    g(<x_i,y_i>, <x_j,y_j>) = sigma(h(x_j,y_j) h(x_i,y_i)^T)
    ... where we override notation h so that it notes for averaged representations of
    all tokens in <x_i, y_i>.

Appendix B ("Training Details of the Forecasting Models")::

    Both trainable logit-based and feature-based forecasting involve learnable
    encoders h to encode input sentences.  For BART0 experiments, we use BART0
    followed by a freshly initialized 2-layer trainable MLP as the encoder h.  For
    FLAN-T5 experiments, we use FLAN-T5_small and a 2-layer MLP as the encoder.  We
    optimize the LM components with a learning rate of 1e-5, and the MLP with a
    learning rate of 1e-4.

This module therefore exposes a single ``EncoderH`` module with **two output modes**:

* ``encode_token_level(x, y)`` -> ``[B, T, d]``: the token-level representation used by
  the trainable logit-change-transfer kernel (Sec. 3.2), where ``T`` is the length of
  the output ``y`` (the representations of the *output* tokens in the final layer);
* ``encode_mean(x, y)`` -> ``[B, d]``: the mean-pooled ("averaged representations of all
  tokens") representation used by the representation-based forecaster (Sec. 3.3).

Reconciliation notes (paper silent -> documented default):
  * ``d``: the paper only says "low-dimensional"; config default is 768, MLP hidden width
    768, 2 layers, dropout 0.0.
  * Optimizer/seeds are not specified by the paper: AdamW (see config) and a fixed seed
    are used by the training scripts.
  * ``T``: interpreted as the output-length axis for ``h`` (Sec. 3.2 defines
    ``f_hat(x) in R^{TV}``, so the token axis of ``h`` is the ``T`` axis); when the
    kernel is formed for a pair with different output lengths, the paper's
    ``Theta_tilde in R^{T x T}`` is materialized as ``R^{T_j x T_i}`` (the only
    consistent rectangular reading, matching ``src/forecasters/logit_based.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

try:  # pragma: no cover - optional import for typing / lazy backbone construction
    from .base_lm import BaseLM, load_base_lm
except Exception:  # pragma: no cover
    BaseLM = Any  # type: ignore

    def load_base_lm(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        raise ImportError(
            "src.modeling.base_lm is required to instantiate an EncoderH backbone"
        )


logger = logging.getLogger("encoder_h")

__all__ = [
    "EncoderH",
    "TwoLayerMLP",
    "load_encoder_h",
    "encoder_h_from_config",
    "BACKBONE_FOR_MODEL",
    "DEFAULT_DIM",
    "DEFAULT_MLP_HIDDEN",
    "DEFAULT_MLP_LAYERS",
    "DEFAULT_LM_LR",
    "DEFAULT_MLP_LR",
]

# --------------------------------------------------------------------------------------
# Defaults (Appendix B + config/config.yaml; the paper leaves the width unspecified)
# --------------------------------------------------------------------------------------
DEFAULT_DIM = 768
DEFAULT_MLP_HIDDEN = 768
DEFAULT_MLP_LAYERS = 2
DEFAULT_LM_LR = 1e-5
DEFAULT_MLP_LR = 1e-4
DEFAULT_DROPOUT = 0.0
DEFAULT_FILENAME = "encoder_h.pt"

#: Appendix B — backbone of ``h`` per base LM used in the experiments.
BACKBONE_FOR_MODEL: Dict[str, str] = {
    "BART0_L": "BART0_L",
    "FLAN-T5_L": "FLAN-T5_small",
    "FLAN-T5_3B": "FLAN-T5_small",
    "FLAN-T5_small": "FLAN-T5_small",
}

_MISSING = object()


# --------------------------------------------------------------------------------------
# MLP head: freshly initialized 2-layer trainable MLP (Appendix B)
# --------------------------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    """Freshly initialized (<=)2-layer MLP mapping backbone hidden size -> ``dim``.

    Appendix B only specifies "a freshly initialized 2-layer trainable MLP"; the
    activation, dropout and width are unspecified, so we use GELU + optional dropout and
    a 768-wide hidden layer (config default).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = DEFAULT_MLP_HIDDEN,
        out_dim: int = DEFAULT_DIM,
        n_layers: int = DEFAULT_MLP_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.n_layers = max(1, int(n_layers))
        self.dropout_p = float(dropout)

        layers: List[nn.Module] = []
        prev = self.in_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(prev, self.hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            prev = self.hidden_dim
        layers.append(nn.Linear(prev, self.out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Fresh initialization of all linear weights ("freshly initialized", App. B)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------------------
# Encoder h
# --------------------------------------------------------------------------------------
class EncoderH(nn.Module):
    """Encoder ``h`` = (trainable) base LM backbone + freshly initialized 2-layer MLP.

    Parameters
    ----------
    backbone:
        A loaded ``BaseLM`` (preferred) or any object exposing one of
        ``token_logits`` / ``encode_token_level`` / ``hidden_states`` / ``final_hidden_states``.
    model_key / backbone_key:
        Registry key used to select the backbone when ``backbone`` is ``None``
        (Appendix B: BART0 for BART0 experiments, FLAN-T5_small for T5 experiments).
    dim:
        Dimensionality ``d`` of the produced representation ``h(x, y)``.
    mlp_hidden, mlp_layers:
        Width / depth of the trainable MLP (paper: 2 layers; width unspecified).
    lm_lr / mlp_lr:
        Per-component learning rates from Appendix B (1e-5 LM, 1e-4 MLP).
    freeze_backbone:
        If ``True``, only the MLP is trained (used for the frozen "Fixed Logit" variant
        and for smoke tests).
    """

    def __init__(
        self,
        backbone: Any = None,
        model_key: str = "BART0_L",
        backbone_key: Optional[str] = None,
        dim: int = DEFAULT_DIM,
        mlp_hidden: int = DEFAULT_MLP_HIDDEN,
        mlp_layers: int = DEFAULT_MLP_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        lm_lr: float = DEFAULT_LM_LR,
        mlp_lr: float = DEFAULT_MLP_LR,
        freeze_backbone: bool = False,
        device: Optional[Any] = None,
        dtype: torch.dtype = torch.float32,
        max_input_len: int = 512,
        max_output_len: int = 64,
        name: str = "encoder_h",
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        self.name = name
        self.model_key = model_key
        self.backbone_key = backbone_key or BACKBONE_FOR_MODEL.get(model_key, model_key)
        self.dim = int(dim)
        self.lm_lr = float(lm_lr)
        self.mlp_lr = float(mlp_lr)
        self.freeze_backbone = bool(freeze_backbone)
        self.max_input_len = int(max_input_len)
        self.max_output_len = int(max_output_len)
        self._dtype = dtype
        self._device = device

        if backbone is None:
            backbone = load_base_lm(
                self.backbone_key,
                device=str(device) if device is not None else "cpu",
                dtype="float32" if dtype in (None, torch.float32) else str(dtype),
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                **backbone_kwargs,
            )
        self.backbone = backbone
        if self.freeze_backbone:
            self.freeze_backbone_parameters()

        in_dim = self._infer_backbone_dim()
        self.mlp = TwoLayerMLP(
            in_dim=in_dim,
            hidden_dim=mlp_hidden,
            out_dim=self.dim,
            n_layers=mlp_layers,
            dropout=dropout,
        )
        if device is not None:
            self.mlp.to(device)

    # ------------------------------------------------------------------- properties
    @property
    def backbone_dim(self) -> int:
        return int(self.mlp.in_dim)

    @property
    def device(self) -> torch.device:
        return next(self.mlp.parameters()).device

    # ------------------------------------------------------------- backbone plumbing
    def _infer_backbone_dim(self) -> int:
        """Best-effort discovery of the backbone hidden size."""
        for attr in ("hidden_size", "d_model", "config"):
            value = getattr(self.backbone, attr, None)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return int(value)
            inner = getattr(value, "hidden_size", None) or getattr(value, "d_model", None)
            if inner:
                return int(inner)
        model = getattr(self.backbone, "model", None)
        cfg = getattr(model, "config", None)
        if cfg is not None:
            inner = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
            if inner:
                return int(inner)
        raise ValueError(
            "Could not infer the backbone hidden size; supply a backbone with a "
            "`hidden_size`/`d_model` attribute or an HF config."
        )

    def set_backbone(self, backbone: Any) -> None:
        self.backbone = backbone
        if self.freeze_backbone:
            self.freeze_backbone_parameters()

    def freeze_backbone_parameters(self) -> None:
        for p in self._iter_backbone_parameters():
            p.requires_grad_(False)
        model = getattr(self.backbone, "model", self.backbone)
        if isinstance(model, nn.Module):
            model.eval()

    def unfreeze_backbone_parameters(self) -> None:
        for p in self._iter_backbone_parameters():
            p.requires_grad_(True)

    def _iter_backbone_parameters(self) -> Iterable[nn.Parameter]:
        backbone = self.backbone
        if isinstance(backbone, nn.Module):
            yield from backbone.parameters()
            return
        inner = getattr(backbone, "model", None)
        if isinstance(inner, nn.Module):
            yield from inner.parameters()

    # -------------------------------------------------------------- hidden extraction
    def backbone_hidden(
        self,
        inputs: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        use_encoder_hidden: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Token-level backbone hidden states ``[B, T, d_backbone]``.

        Sec. 3.2: "We implement h with a trainable LM and extract its representation of
        output tokens in the final layer as h(x, y)". ``use_encoder_hidden=True``
        optionally returns encoder states instead (used for the frozen head-only kernel
        of the "Fixed Logit" variant, where the gradient w.r.t. the LM head weights is
        exactly the representation of ``x`` before the LM head).
        """
        backbone = self.backbone
        inputs = list(inputs)
        if targets is not None:
            targets = list(targets)

        # 1) Preferred: a BaseLM-style logits call that can also return hidden states.
        for method_name in ("token_logits", "forward_logits", "logits"):
            method = getattr(backbone, method_name, None)
            if method is None:
                continue
            out: Any = None
            for call_kwargs in (
                {"return_hidden": True, "use_encoder_hidden": use_encoder_hidden},
                {"return_hidden": True},
                {},
            ):
                try:
                    out = method(inputs, targets, **call_kwargs, **kwargs)
                    break
                except TypeError:
                    continue
                except Exception:  # pragma: no cover - genuine model error
                    out = None
                    break
            hidden = self._hidden_from_output(out)
            if hidden is not None:
                return hidden

        # 2) Explicit hidden-state encoders.
        for method_name in (
            "encode_token_level",
            "encode_token",
            "hidden_states",
            "final_hidden_states",
            "encode",
        ):
            method = getattr(backbone, method_name, None)
            if method is None:
                continue
            out = None
            for call_args in ((inputs, targets), (inputs,)):
                try:
                    out = method(*call_args)
                    break
                except TypeError:
                    continue
                except Exception:  # pragma: no cover
                    out = None
                    break
            hidden = self._hidden_from_output(out)
            if hidden is not None:
                return hidden

        raise AttributeError(
            "The provided backbone exposes none of token_logits / encode_token_level / "
            "hidden_states / final_hidden_states; cannot extract token-level representations."
        )

    @staticmethod
    def _hidden_from_output(out: Any) -> Optional[torch.Tensor]:
        if out is None:
            return None
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, Mapping):
            for key in (
                "hidden",
                "hidden_states",
                "hidden_state",
                "encoder_hidden",
                "last_hidden_state",
                "repr",
                "representation",
            ):
                value = out.get(key, _MISSING)
                if isinstance(value, torch.Tensor):
                    return value
            return None
        if isinstance(out, (tuple, list)):
            for item in out[1:]:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    return item
            return None
        return None

    # -------------------------------------------------------------- forward / encode
    def forward(
        self,
        inputs: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        hidden: Optional[torch.Tensor] = None,
        mean_pool: bool = False,
        mask: Optional[torch.Tensor] = None,
        use_encoder_hidden: bool = False,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode examples with ``h``.

        Either pass raw ``inputs``/``targets`` strings, or a precomputed ``hidden`` tensor
        ``[B, T, d_backbone]`` (useful when backbone states are cached). Returns
        ``[B, T, d]`` for ``mean_pool=False`` and ``[B, d]`` otherwise.
        """
        if hidden is None:
            if inputs is None:
                raise ValueError("Either `inputs` or `hidden` must be provided.")
            inputs = list(inputs)
            bs = int(batch_size or max(1, len(inputs)))
            hidden = self.backbone_hidden(
                inputs,
                targets,
                use_encoder_hidden=use_encoder_hidden,
                batch_size=bs,
                **kwargs,
            )
        hidden = self._as_tensor(hidden)
        if self._device is not None:
            hidden = hidden.to(self._device)
        param = next(self.mlp.parameters())
        hidden = hidden.to(device=param.device, dtype=param.dtype)
        out = self.mlp(hidden)
        if mean_pool:
            return mean_pool_representation(out, mask)
        return out

    def encode(
        self,
        inputs: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        mean_pool: bool = False,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            inputs=inputs, targets=targets, mean_pool=mean_pool,
            batch_size=batch_size, **kwargs
        )

    def encode_token_level(
        self,
        inputs: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Token-level ``h(x, y)`` in ``R^{T x d}`` for the logit-change kernel (Sec. 3.2)."""
        return self.forward(
            inputs=inputs, targets=targets, mean_pool=False, batch_size=batch_size, **kwargs
        )

    def encode_mean(
        self,
        inputs: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        mask: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Mean-pooled ``h(x, y)`` in ``R^d`` for the representation model (Sec. 3.3).

        Sec. 3.3: "we override notation h so that it notes for averaged representations of
        all tokens in <x_i, y_i>" — the average is taken over the token axis of the
        token-level encoding of output tokens (Sec. 3.2).
        """
        return self.forward(
            inputs=inputs, targets=targets, mean_pool=True, mask=mask,
            batch_size=batch_size, **kwargs
        )

    # ------------------------------------------------------------------ optim / io
    def param_groups(self) -> List[Dict[str, Any]]:
        """LM components at ``lm_lr`` and MLP at ``mlp_lr`` (Appendix B: 1e-5 / 1e-4)."""
        lm_params = [p for p in self._iter_backbone_parameters() if p.requires_grad]
        mlp_params = [p for p in self.mlp.parameters() if p.requires_grad]
        groups: List[Dict[str, Any]] = []
        if lm_params:
            groups.append({"params": lm_params, "lr": self.lm_lr, "name": "lm"})
        if mlp_params:
            groups.append({"params": mlp_params, "lr": self.mlp_lr, "name": "mlp"})
        return groups

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> "EncoderH":  # noqa: D102
        super().train(mode)
        if self.freeze_backbone:
            model = getattr(self.backbone, "model", self.backbone)
            if isinstance(model, nn.Module):
                model.eval()
        return self

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_key": self.model_key,
            "backbone_key": self.backbone_key,
            "backbone_dim": self.backbone_dim,
            "dim": self.dim,
            "mlp_hidden": self.mlp.hidden_dim,
            "mlp_layers": self.mlp.n_layers,
            "dropout": self.mlp.dropout_p,
            "lm_lr": self.lm_lr,
            "mlp_lr": self.mlp_lr,
            "freeze_backbone": self.freeze_backbone,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
        }

    def save(self, path: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        """Persist the trainable MLP + metadata (the backbone is reloaded by key)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload: Dict[str, Any] = {"config": self.config(), "mlp": self.mlp.state_dict()}
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        return path

    def load_state(self, path: str, strict: bool = False) -> "EncoderH":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping) and "mlp" in payload:
            self.mlp.load_state_dict(payload["mlp"], strict=strict)
        else:  # raw MLP state dict
            self.mlp.load_state_dict(payload, strict=strict)
        return self

    # ------------------------------------------------------------------ static utils
    @staticmethod
    def _as_tensor(hidden: Any) -> torch.Tensor:
        if isinstance(hidden, torch.Tensor):
            return hidden
        if isinstance(hidden, Mapping):
            for key in ("hidden", "hidden_states", "last_hidden_state"):
                value = hidden.get(key)
                if isinstance(value, torch.Tensor):
                    return value
        return torch.as_tensor(hidden)


# --------------------------------------------------------------------------------------
# Mean pooling (Sec. 3.3: "averaged representations of all tokens")
# --------------------------------------------------------------------------------------
def mean_pool_representation(
    hidden: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Masked mean over the token axis.

    ``hidden``: ``[B, T, d]`` (or ``[T, d]``); ``mask``: ``[B, T]`` with 1 for real tokens
    (padding excluded from the average). When ``mask`` is omitted, non-zero activations
    are used as an implicit padding mask (falling back to a plain mean if all-zero).
    """
    squeeze = hidden.dim() == 2
    if squeeze:
        hidden = hidden.unsqueeze(0)
    dtype = hidden.dtype
    if mask is None:
        mask = (hidden.abs().sum(dim=-1) > 0).to(dtype)
        if mask.sum(dim=-1).min() == 0:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=dtype)
    mask = mask.to(device=hidden.device, dtype=dtype)
    while mask.dim() < hidden.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    pooled = (hidden * mask).sum(dim=1) / denom
    return pooled.squeeze(0) if squeeze else pooled


# --------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------
def encoder_h_from_config(
    config: Optional[Mapping[str, Any]] = None,
    model_key: str = "BART0_L",
    backbone: Any = None,
    **overrides: Any,
) -> EncoderH:
    """Build an ``EncoderH`` from ``config.yaml``'s ``encoder_h`` section.

    Appendix B behaviour is encoded in the defaults: the backbone of ``h`` is BART0 for
    BART0 experiments and FLAN-T5_small for FLAN-T5 experiments (``BACKBONE_FOR_MODEL``);
    LM learning rate 1e-5, MLP learning rate 1e-4.
    """
    section: Dict[str, Any] = {}
    model_cfg: Dict[str, Any] = {}
    if config:
        section = dict(config.get("encoder_h", {}) or {})
        model_cfg = dict((config.get("models", {}) or {}).get(model_key, {}) or {})

    backbone_map = section.get("backbone", {}) or {}
    backbone_key = overrides.pop("backbone_key", None)
    if backbone_key is None:
        backbone_key = backbone_map.get(model_key) or BACKBONE_FOR_MODEL.get(model_key, model_key)

    params: Dict[str, Any] = {
        "model_key": model_key,
        "backbone_key": backbone_key,
        "dim": section.get("repr_dim", section.get("dim", DEFAULT_DIM)),
        "mlp_hidden": section.get("mlp_hidden", DEFAULT_MLP_HIDDEN),
        "mlp_layers": section.get("mlp_layers", DEFAULT_MLP_LAYERS),
        "dropout": section.get("dropout", DEFAULT_DROPOUT),
        "lm_lr": section.get("lm_lr", DEFAULT_LM_LR),
        "mlp_lr": section.get("mlp_lr", DEFAULT_MLP_LR),
        "freeze_backbone": section.get("freeze_backbone", False),
        "device": section.get("device", (config or {}).get("device")),
        "max_input_len": section.get(
            "max_input_len", model_cfg.get("max_input_len", 512)
        ),
        "max_output_len": section.get(
            "max_output_len", model_cfg.get("max_output_len", 64)
        ),
    }
    params.update(overrides)
    if params.get("device") is None:
        params["device"] = "cpu"
    if params["device"] == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU for h.")
        params["device"] = "cpu"
    return EncoderH(backbone=backbone, **params)


def load_encoder_h(
    model_key: str = "BART0_L",
    config: Optional[Mapping[str, Any]] = None,
    backbone: Any = None,
    **overrides: Any,
) -> EncoderH:
    """Load the encoding function ``h`` for a given base LM.

    ``backbone`` may be supplied directly (e.g. the already-loaded ``f_0``); otherwise a
    fresh backbone is instantiated from ``src.modeling.base_lm`` using the backbone key
    selected per Appendix B.
    """
    return encoder_h_from_config(
        config=config, model_key=model_key, backbone=backbone, **overrides
    )


# --------------------------------------------------------------------------------------
# CLI / self-test
# --------------------------------------------------------------------------------------
class _DummyBackbone(nn.Module):
    """Tiny offline stand-in exposing the ``token_logits`` contract (self-test only)."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def token_logits(
        self,
        inputs: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        return_hidden: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        batch = len(inputs)
        t = max(1, len(targets[0].split())) if targets else 4
        hidden = torch.randn(batch, t, self.hidden_size)
        return {"logits": torch.randn(batch, t, self.vocab_size), "hidden": hidden}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encoding function h (Sec. 3.2/3.3, Appendix B)"
    )
    parser.add_argument("--config", type=str, default=None, help="path to config/config.yaml")
    parser.add_argument("--model", type=str, default="BART0_L", help="base LM registry key")
    parser.add_argument("--dim", type=int, default=None, help="representation dimensionality d")
    parser.add_argument("--mlp-hidden", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text", type=str, default="Say hello in French.", help="example input")
    parser.add_argument("--target", type=str, default="Bonjour.", help="example output")
    parser.add_argument("--self-test", action="store_true", help="offline dummy-backbone test")
    parser.add_argument("--out", type=str, default=None, help="optional checkpoint path")
    return parser.parse_args(argv)


def _self_test() -> int:
    torch.manual_seed(0)
    backbone = _DummyBackbone()
    enc = EncoderH(backbone=backbone, dim=16, mlp_hidden=16, mlp_layers=2)
    inputs = ["a b c", "d e"]
    targets = ["x y", "z"]
    token = enc.encode_token_level(inputs, targets)
    mean = enc.encode_mean(inputs, targets)
    assert token.dim() == 3 and token.shape[-1] == 16, tuple(token.shape)
    assert mean.dim() == 2 and mean.shape[-1] == 16, tuple(mean.shape)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    pooled = mean_pool_representation(torch.ones(2, 3, 5), mask)
    assert pooled.shape == (2, 5)
    assert torch.allclose(pooled[0], torch.tensor([1.0] * 5)), pooled[0]
    groups = enc.param_groups()
    assert groups and any(g["lr"] == DEFAULT_MLP_LR for g in groups)
    logger.info(
        "self-test passed: token=%s mean=%s groups=%d",
        tuple(token.shape),
        tuple(mean.shape),
        len(groups),
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()

    config: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load config %s: %s", args.config, exc)

    overrides: Dict[str, Any] = {}
    if args.dim is not None:
        overrides["dim"] = args.dim
    if args.mlp_hidden is not None:
        overrides["mlp_hidden"] = args.mlp_hidden
    if args.mlp_layers is not None:
        overrides["mlp_layers"] = args.mlp_layers
    if args.device is not None:
        overrides["device"] = args.device

    enc = load_encoder_h(model_key=args.model, config=config, **overrides)
    logger.info("encoder config: %s", json.dumps(enc.config(), indent=2))
    with torch.no_grad():
        token = enc.encode_token_level([args.text], [args.target])
        mean = enc.encode_mean([args.text], [args.target])
    logger.info("h(x,y) token-level shape: %s", tuple(token.shape))
    logger.info("h(x,y) mean shape: %s", tuple(mean.shape))
    if args.out:
        logger.info("saved to %s", enc.save(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
