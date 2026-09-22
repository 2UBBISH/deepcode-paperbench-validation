"""Wrapper around the encoder-decoder PTLMs used in the paper.

The wrapper exposes exactly the quantities the reproduction needs:

* ``predict``          -- generation, used for Exact Match / edit success rate.
* ``teacher_forced_logits`` -- pre-softmax logits ``f(x) in R^{T x V}`` of the gold
  target tokens, used by the logit-change forecasting models (Sec. 3.2).
* ``encode``           -- per-token representations ``h(x, y)`` of encoder and
  decoder tokens, used by both trainable forecasting models (Sec. 3.2, 3.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ModelSpec
from ..evaluation.metrics import exact_match
from ..data.types import Example


@dataclass
class LogitCache:
    """Logits of the gold target tokens of one example: ``[T, V]`` (V may be pruned)."""

    values: np.ndarray        # [T, V] float32
    token_ids: np.ndarray     # [T] int64, gold token ids
    vocab_ids: Optional[np.ndarray] = None  # [V] original vocab index of each column


class Seq2SeqLM:
    """A HF encoder-decoder LM (BART0, FLAN-T5, ...) with the helpers above."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cpu",
        max_input_len: int = 512,
        max_target_len: int = 32,
        dtype: str = "float32",
        hf_name: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.spec = spec
        self.device = device
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model_name = hf_name or spec.hf_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.hf_name = model_name

    # ----------------------------------------------------------------------------------
    # basics
    # ----------------------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(getattr(self.model.config, "vocab_size"))

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: str) -> "Seq2SeqLM":
        self.device = device
        self.model.to(device)
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ----------------------------------------------------------------------------------
    # generation / exact match
    # ----------------------------------------------------------------------------------
    def _encode_inputs(self, inputs: Sequence[str]):
        return self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        )

    def predict(
        self,
        inputs: Sequence[str],
        batch_size: int = 16,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
    ) -> List[str]:
        """Greedy (``num_beams=1``) generation, as used for Exact Match."""
        import torch

        max_new_tokens = max_new_tokens or self.max_target_len
        preds: List[str] = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = list(inputs[start:start + batch_size])
                enc = self._encode_inputs(batch).to(self.device)
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
                preds.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        if was_training:
            self.model.train()
        return preds

    def exact_match(self, examples: Sequence[Example], batch_size: int = 16) -> float:
        preds = self.predict([e.input for e in examples], batch_size=batch_size)
        return exact_match(preds, [e.target for e in examples])

    def correctness(self, examples: Sequence[Example], batch_size: int = 16) -> List[bool]:
        preds = self.predict([e.input for e in examples], batch_size=batch_size)
        return [exact_match([p], [e.target]) > 0.5 for p, e in zip(preds, examples)]

    # ----------------------------------------------------------------------------------
    # logits / hidden states
    # ----------------------------------------------------------------------------------
    def teacher_forced_logits(
        self,
        inputs: Sequence[str],
        targets: Sequence[str],
        batch_size: int = 8,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Pre-softmax logits of the gold target tokens.

        Returns one ``(logits[ T, V ], label_ids[ T ])`` pair per example, where
        ``T`` is the (truncated) target length.
        """
        import torch

        was_training = self.model.training
        self.model.eval()
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch_x = list(inputs[start:start + batch_size])
                batch_y = list(targets[start:start + batch_size])
                enc = self._encode_inputs(batch_x).to(self.device)
                labels = self.tokenizer(
                    batch_y,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_target_len,
                ).input_ids.to(self.device)
                out = self.model(**enc, labels=labels)
                logits = out.logits.float().cpu().numpy()          # [B, T, V]
                label_ids = labels.cpu().numpy()
                for b in range(logits.shape[0]):
                    results.append((logits[b], label_ids[b]))
        if was_training:
            self.model.train()
        return results

    def encode(
        self,
        inputs: Sequence[str],
        targets: Sequence[str],
        batch_size: int = 8,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Hidden representations of the last layer.

        Returns ``(encoder_states[L_x, H], decoder_states[L_y, H])`` per example.
        ``decoder_states`` are the hidden states aligned with the gold target
        tokens, i.e. the representation used by the logit-change kernel.
        """
        import torch

        was_training = self.model.training
        self.model.eval()
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch_x = list(inputs[start:start + batch_size])
                batch_y = list(targets[start:start + batch_size])
                enc = self._encode_inputs(batch_x).to(self.device)
                labels = self.tokenizer(
                    batch_y,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_target_len,
                ).input_ids.to(self.device)
                out = self.model(**enc, labels=labels, output_hidden_states=True)
                enc_states = out.encoder_hidden_states[-1].float().cpu().numpy()
                dec_states = out.decoder_hidden_states[-1].float().cpu().numpy()
                enc_mask = enc["attention_mask"].cpu().numpy()
                for b in range(len(batch_x)):
                    length = int(enc_mask[b].sum())
                    target_len = int((labels[b] != self._pad_token_id()).sum())
                    results.append((enc_states[b, :length], dec_states[b, :target_len]))
        if was_training:
            self.model.train()
        return results

    def _pad_token_id(self) -> int:
        pad = self.tokenizer.pad_token_id
        return 0 if pad is None else int(pad)

    # ----------------------------------------------------------------------------------
    # parameter snapshots (f_0 -> f_i resets)
    # ----------------------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        import copy

        return {k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()}

    def restore(self, snapshot: Dict[str, object]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in snapshot.items()})

    def clone(self) -> "Seq2SeqLM":
        clone = Seq2SeqLM.__new__(Seq2SeqLM)
        import copy

        clone.spec = self.spec
        clone.device = self.device
        clone.max_input_len = self.max_input_len
        clone.max_target_len = self.max_target_len
        clone.torch_dtype = self.torch_dtype
        clone.tokenizer = self.tokenizer
        clone.model = copy.deepcopy(self.model)
        clone.hf_name = self.hf_name
        return clone

    @classmethod
    def from_config(cls, cfg) -> "Seq2SeqLM":
        from ..utils import resolve_device

        return cls(
            cfg.model_spec(),
            device=resolve_device(cfg.device),
            max_input_len=cfg.max_input_len,
            max_target_len=cfg.max_target_len,
        )
