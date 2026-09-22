"""Energy adapter ``g_theta`` for BBox-Adapter.

The adapter maps a ``(question, candidate answer)`` text pair to a *scalar
energy* :math:`g_\\theta(x, y)`.  Formally, BBox-Adapter frames black-box LLM
adaptation as sampling from a specialised energy-based sequence model

.. math::

    p_\\theta(y \\mid x) = p_{\\mathrm{LLM}}(y \\mid x)
        \\frac{\\exp(g_\\theta(x, y))}{Z_\\theta(x)}

where :math:`p_{\\mathrm{LLM}}` stays fixed (the black-box model) and only the
adapter :math:`g_\\theta` is trained (Section 3.1).

Implementation follows Appendix H.2 / Appendix E / Section 4.1:

* ``microsoft/deberta-v3-base``  (86M params, "0.1B") -- StrategyQA, GSM8K,
  ScienceQA and ToxiGen;
* ``microsoft/deberta-v3-large`` (304M params, "0.3B") -- same datasets;
* ``bert-base-cased`` (110M params) -- TruthfulQA.

A single scalar head sits on top of the pooled encoder representation.  The
backbone is *randomly initialised* from its pretrained weights (it is a small
open model, not the black-box LLM) and the head is freshly initialised before
online adaptation.

Nothing in this module ever touches the black-box LLM's log-probabilities,
hidden states or gradients: it consumes raw text only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "ADAPTER_BACKBONES",
    "SIZE_BACKBONES",
    "EnergyModelConfig",
    "EnergyModel",
    "resolve_backbone",
    "build_energy_model",
    "format_pair",
    "batch_pairs",
]


# --------------------------------------------------------------------------- #
# Backbone resolution
# --------------------------------------------------------------------------- #

#: Backbone used per dataset, following Appendix H.2 / E.  ``size`` overrides it.
ADAPTER_BACKBONES: Dict[str, str] = {
    "strategyqa": "microsoft/deberta-v3-base",
    "gsm8k": "microsoft/deberta-v3-base",
    "scienceqa": "microsoft/deberta-v3-base",
    "toxigen": "microsoft/deberta-v3-base",
    "truthfulqa": "bert-base-cased",
}

#: Paper's parameter-count aliases -> HuggingFace checkpoint.
SIZE_BACKBONES: Dict[str, str] = {
    "0.1b": "microsoft/deberta-v3-base",
    "0.3b": "microsoft/deberta-v3-large",
    "86m": "microsoft/deberta-v3-base",
    "110m": "bert-base-cased",
    "304m": "microsoft/deberta-v3-large",
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}

#: Approximate parameter counts (in billions) per checkpoint, for reporting.
_BACKBONE_PARAMS_B = {
    "microsoft/deberta-v3-base": 0.086,
    "microsoft/deberta-v3-large": 0.304,
    "bert-base-cased": 0.110,
}


def resolve_backbone(
    dataset: Optional[str] = None,
    size: Optional[str] = None,
    backbone: Optional[str] = None,
) -> str:
    """Resolve the adapter backbone checkpoint.

    Precedence: explicit ``backbone`` > ``size`` alias (``0.1b``/``0.3b``) >
    per-dataset default.  ``size=None``/``"none"`` means "use the dataset
    default" (which is deberta-v3-base for every dataset except TruthfulQA).
    """

    if backbone:
        return backbone
    size_key = str(size).strip().lower() if size is not None else ""
    if size_key and size_key in SIZE_BACKBONES:
        # TruthfulQA is always paired with bert-base-cased in the paper, except
        # when the user explicitly asks for a parameter-size variant.
        if size_key in {"0.1b", "0.3b", "86m", "304m"}:
            return SIZE_BACKBONES[size_key]
        return SIZE_BACKBONES[size_key]
    if dataset is None:
        return ADAPTER_BACKBONES["strategyqa"]
    key = str(dataset).strip().lower().replace("-", "").replace("_", "")
    for name, ckpt in ADAPTER_BACKBONES.items():
        if name.replace("-", "") == key:
            return ckpt
    return ADAPTER_BACKBONES["strategyqa"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class EnergyModelConfig:
    """Configuration of the scalar energy adapter ``g_theta``."""

    #: HuggingFace encoder checkpoint.
    backbone: str = "microsoft/deberta-v3-base"
    #: Pooling strategy for the encoder output: ``"cls"``, ``"mean"`` or ``"max"``.
    pooling: str = "cls"
    #: Hidden size of the scalar head (0 -> a single Linear layer is used).
    head_hidden: int = 0
    #: Dropout applied before the scalar projection.
    dropout: float = 0.0
    #: Freeze the backbone encoder and train the head only (not used by default).
    freeze_backbone: bool = False
    #: Maximum token length for the (question, answer) pair.
    max_length: int = 512
    #: Add a post-hoc output scaling (energy temperature). 1.0 keeps g in nats.
    output_scale: float = 1.0
    #: Initialisation std of the scalar head (zero-init would stall training).
    head_init_std: float = 0.02
    #: Dataset this adapter targets (informational / used for checkpoint naming).
    dataset: Optional[str] = None
    #: Paper alias for the adapter size, e.g. ``"0.1b"``.
    size: Optional[str] = None
    #: Extra HF kwargs (e.g. ``{"torch_dtype": "float32"}``).
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_backbone(self) -> str:
        return resolve_backbone(
            dataset=self.dataset, size=self.size, backbone=self.backbone
        )


# --------------------------------------------------------------------------- #
# Text formatting / tokenisation helpers
# --------------------------------------------------------------------------- #


def format_pair(question: str, answer: str, sep: str = "\n") -> str:
    """Render a ``(question, candidate answer)`` pair as a single string.

    ``sep`` defaults to a newline: ``"Question: <x>\\nAnswer: <y>"``.
    """

    question = "" if question is None else str(question)
    answer = "" if answer is None else str(answer)
    return f"Question: {question.strip()}{sep}Answer: {answer.strip()}"


def batch_pairs(
    questions: Sequence[str],
    answers: Sequence[str],
    sep: str = "\n",
) -> List[str]:
    """Vectorised :func:`format_pair`."""

    if len(questions) != len(answers):
        raise ValueError(
            f"questions/answers length mismatch: {len(questions)} vs {len(answers)}"
        )
    return [format_pair(q, a, sep=sep) for q, a in zip(questions, answers)]


# --------------------------------------------------------------------------- #
# The energy model
# --------------------------------------------------------------------------- #


class EnergyModel(nn.Module):
    """Scalar energy adapter :math:`g_\\theta(x, y)`.

    Parameters
    ----------
    config:
        :class:`EnergyModelConfig` describing the backbone and head.
    tokenizer:
        Optional pre-built tokenizer; otherwise loaded from the backbone path.

    The forward pass returns one *scalar energy per pair*, shape ``[B]``.
    Regularisation (``alpha * E[g^2]``) is applied by the loss, not here, see
    :mod:`bbox_adapter.adapter.regularizer`.
    """

    def __init__(
        self,
        config: Optional[EnergyModelConfig] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config or EnergyModelConfig()
        self.backbone_name = self.config.resolved_backbone()

        encoder_cls, tokenizer_cls = _hf_classes()
        extra = dict(self.config.extra or {})
        # ``use_safetensors`` / dtype kwargs are forwarded when provided.
        self.encoder = encoder_cls.from_pretrained(self.backbone_name, **extra)
        self.hidden_size = int(getattr(self.encoder.config, "hidden_size"))

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = tokenizer_cls.from_pretrained(self.backbone_name)

        if self.config.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        head_hidden = int(self.config.head_hidden)
        dropout = float(self.config.dropout)
        if head_hidden > 0:
            self.head: nn.Module = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, head_hidden),
                nn.Tanh(),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden_size, 1))

        self._init_head()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #

    def _init_head(self) -> None:
        """Randomly initialise the scalar head (matches 'randomly initialized')."""

        std = float(self.config.head_init_std)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_pretrained(
        cls,
        path_or_dir: str,
        config: Optional[EnergyModelConfig] = None,
        map_location: str = "cpu",
    ) -> "EnergyModel":
        """Load a previously adapted adapter from disk."""

        cfg_path = os.path.join(path_or_dir, "energy_config.pt")
        state_path = os.path.join(path_or_dir, "energy_model.pt")
        saved_cfg = None
        if os.path.isfile(cfg_path):
            saved_cfg = torch.load(cfg_path, map_location="cpu")
        if config is None:
            config = EnergyModelConfig(**(saved_cfg or {}))
        elif saved_cfg:
            merged = dict(saved_cfg)
            merged.update({k: v for k, v in config.__dict__.items() if v is not None})
            config = EnergyModelConfig(**merged)

        model = cls(config)
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location=map_location)
            missing, unexpected = model.load_state_dict(state, strict=False)
            model._missing_keys = list(missing)
            model._unexpected_keys = list(unexpected)
        return model

    def save_pretrained(self, path_or_dir: str) -> None:
        """Persist the adapter (weights + config) to ``path_or_dir``."""

        os.makedirs(path_or_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path_or_dir, "energy_model.pt"))
        torch.save(self.config.__dict__, os.path.join(path_or_dir, "energy_config.pt"))
        try:  # convenience for re-loading the tokenizer alone
            self.tokenizer.save_pretrained(path_or_dir)
        except Exception:  # pragma: no cover - tokenizer saving is best effort
            pass

    # ------------------------------------------------------------------ #
    # tokenisation / forward
    # ------------------------------------------------------------------ #

    def tokenize(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenise ``(question, answer)`` pairs into encoder inputs."""

        texts = batch_pairs(questions, answers)
        max_length = int(max_length or self.config.max_length)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return dict(enc)

    def _encoder_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Drop inputs the encoder does not accept (e.g. DeBERTa-v3 has no token_type_ids)."""

        accepted = set(getattr(self.encoder, "forward").__code__.co_varnames)  # type: ignore[union-attr]
        accepted |= {"input_ids", "attention_mask", "token_type_ids", "inputs_embeds"}
        kwargs: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key in {"input_ids", "attention_mask", "token_type_ids", "inputs_embeds"}:
                if key in accepted:
                    kwargs[key] = value
        return kwargs

    def pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool ``[B, T, H]`` encoder states to ``[B, H]``."""

        mode = str(self.config.pooling).lower()
        if mode == "cls":
            return last_hidden_state[:, 0, :]
        if mode == "max":
            if attention_mask is None:
                return last_hidden_state.max(dim=1).values
            mask = attention_mask.unsqueeze(-1).bool()
            return last_hidden_state.masked_fill(~mask, float("-inf")).max(dim=1).values
        # mean pooling (mask-aware)
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar energies of shape ``[B]``."""

        batch: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        kwargs = self._encoder_kwargs(batch)
        outputs = self.encoder(**kwargs)
        hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        pooled = self.pool(hidden, kwargs.get("attention_mask"))
        energy = self.head(pooled).squeeze(-1)
        return energy * float(self.config.output_scale)

    # ------------------------------------------------------------------ #
    # convenience scoring API (used by losses / beam search / buffers)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def energy(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        batch_size: int = 64,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Score pairs without gradients; returns ``[N]`` energies (CPU tensor)."""

        self.eval()
        device = self.device
        out: List[torch.Tensor] = []
        n = len(questions)
        for start in range(0, n, max(1, batch_size)):
            q_batch = questions[start : start + batch_size]
            a_batch = answers[start : start + batch_size]
            batch = self.tokenize(q_batch, a_batch, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            out.append(self.forward(**batch).detach().cpu())
        if not out:
            return torch.zeros(0)
        return torch.cat(out, dim=0)

    def score_pairs(
        self,
        pairs: Iterable[Tuple[str, str]],
        batch_size: int = 64,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Gradient-enabled scoring over an iterable of ``(question, answer)`` pairs."""

        pairs = list(pairs)
        questions = [p[0] for p in pairs]
        answers = [p[1] for p in pairs]
        return self.score_batch(questions, answers, batch_size, max_length=max_length)

    def score_batch(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        batch_size: int = 64,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Gradient-enabled batched scoring returning concatenated ``[N]`` energies.

        The concatenation preserves the autograd graph so the ranking-NCE loss
        (Eq. 2) can be differentiated w.r.t. ``theta`` (Eq. 3).
        """

        device = self.device
        chunks: List[torch.Tensor] = []
        n = len(questions)
        for start in range(0, n, max(1, batch_size)):
            q_batch = list(questions[start : start + batch_size])
            a_batch = list(answers[start : start + batch_size])
            batch = self.tokenize(q_batch, a_batch, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            chunks.append(self.forward(**batch))
        if not chunks:
            return torch.zeros(0, device=device)
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_summary(self) -> Dict[str, Any]:
        total = self.num_parameters()
        return {
            "backbone": self.backbone_name,
            "dataset": self.config.dataset,
            "size_alias": self.config.size,
            "pooling": self.config.pooling,
            "head_hidden": self.config.head_hidden,
            "params_total": total,
            "params_billions": total / 1e9,
            "params_head": sum(
                p.numel() for p in self.head.parameters() if p.requires_grad
            ),
            "paper_backbone_params_b": _BACKBONE_PARAMS_B.get(self.backbone_name),
        }

    def train_mode_backbone(self, trainable: bool) -> None:
        """Enable/disable fine-tuning of the encoder (head is always trained)."""

        for param in self.encoder.parameters():
            param.requires_grad_(trainable)


# --------------------------------------------------------------------------- #
# HF imports kept lazy so the module imports without transformers installed
# --------------------------------------------------------------------------- #


def _hf_classes():
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Building an EnergyModel requires the `transformers` package "
            "(pip install transformers). Original error: %s" % exc
        )
    return AutoModel, AutoTokenizer


def build_energy_model(
    dataset: Optional[str] = None,
    size: Optional[str] = None,
    backbone: Optional[str] = None,
    pooling: Optional[str] = None,
    max_length: int = 512,
    **kwargs: Any,
) -> EnergyModel:
    """Factory used by the training scripts / configs.

    Example
    -------
    >>> model = build_energy_model(dataset="strategyqa", size="0.1b")
    >>> model.backbone_name
    'microsoft/deberta-v3-base'
    """

    resolved = resolve_backbone(dataset=dataset, size=size, backbone=backbone)
    config = EnergyModelConfig(
        backbone=resolved,
        pooling=pooling or ("cls" if "bert" in resolved else "mean"),
        max_length=max_length,
        dataset=dataset,
        size=size,
        **kwargs,
    )
    return EnergyModel(config)
