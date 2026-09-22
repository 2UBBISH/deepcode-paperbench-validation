"""The learnable encoder ``h`` of the two trainable forecasting models.

Sec. 3.2/3.3 and Appendix B: "For BART0 experiments, we use BART0 followed by a
freshly initialized 2-layer trainable MLP as the encoder h.  For FLAN-T5
experiments, we use FLAN-T5_small and a 2-layer MLP as the encoder.  We optimize
the LM components with a learning rate of 1e-5, and the MLP with a learning rate
of 1e-4."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EncoderOutputs:
    """Outputs of ``h`` for a batch of examples."""

    token_reps: "object"        # [B, T, d] torch tensor: representation of every gold target token
    pooled: "object"            # [B, d] torch tensor: averaged representation of <x, y>
    mask: "object"              # [B, T] bool tensor


def build_encoder(
    model_name: str,
    device: str = "cpu",
    hidden_dim: int = 512,
    trainable_lm: bool = True,
    dropout: float = 0.0,
    max_input_len: int = 512,
    max_target_len: int = 32,
):
    """Instantiate a :class:`PairEncoder` (imported lazily to keep torch optional)."""
    return PairEncoder(
        model_name=model_name,
        device=device,
        hidden_dim=hidden_dim,
        trainable_lm=trainable_lm,
        dropout=dropout,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )


class PairEncoder:
    """``h: (x, y) -> R^{T x d}`` (per-token) and ``R^d`` (pooled).

    Implemented with a HF encoder-decoder LM: the decoder hidden states of the
    gold target tokens give the per-token representation used by the logit-change
    kernel, and the average of all encoder/decoder token states gives the pooled
    representation used by the representation-based forecaster.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        hidden_dim: int = 512,
        trainable_lm: bool = True,
        dropout: float = 0.0,
        max_input_len: int = 512,
        max_target_len: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        for param in self.lm.parameters():
            param.requires_grad_(trainable_lm)
        if not trainable_lm:
            self.lm.eval()
        hidden = int(self.lm.config.d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        self.hidden_dim = hidden_dim

    # ----------------------------------------------------------------------------------
    def to(self, device: str) -> "PairEncoder":
        self.device = device
        self.lm.to(device)
        self.mlp.to(device)
        return self

    def parameters(self):
        return list(self.lm.parameters()) + list(self.mlp.parameters())

    def parameter_groups(self, lm_lr: float, mlp_lr: float) -> List[Dict]:
        """Optimizer groups: LM components at ``lm_lr``, MLP at ``mlp_lr``."""
        return [
            {"params": [p for p in self.lm.parameters() if p.requires_grad], "lr": lm_lr},
            {"params": list(self.mlp.parameters()), "lr": mlp_lr},
        ]

    # ----------------------------------------------------------------------------------
    def _tokenize(self, inputs: Sequence[str], targets: Sequence[str]):
        enc = self.tokenizer(
            list(inputs), return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_len
        ).to(self.device)
        labels = self.tokenizer(
            list(targets), return_tensors="pt", padding=True, truncation=True, max_length=self.max_target_len
        ).input_ids.to(self.device)
        return enc, labels

    def forward(self, inputs: Sequence[str], targets: Sequence[str]) -> EncoderOutputs:
        torch = self.torch
        enc, labels = self._tokenize(inputs, targets)
        out = self.lm(
            **enc, labels=labels, output_hidden_states=True, use_cache=False
        )
        dec_states = out.decoder_hidden_states[-1]                       # [B, T, H]
        enc_states = out.encoder_hidden_states[-1]                       # [B, L, H]
        token_reps = self.mlp(dec_states)                                # [B, T, d]
        enc_reps = self.mlp(enc_states)                                  # [B, L, d]
        mask = labels != (self.tokenizer.pad_token_id or 0)
        mask_f = mask.unsqueeze(-1).to(enc_reps.dtype)
        enc_mask = enc["attention_mask"].unsqueeze(-1).to(enc_reps.dtype)
        pooled_dec = (token_reps * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        pooled_enc = (enc_reps * enc_mask).sum(1) / enc_mask.sum(1).clamp(min=1.0)
        pooled = 0.5 * (pooled_dec + pooled_enc)
        return EncoderOutputs(token_reps=token_reps, pooled=pooled, mask=mask)

    __call__ = forward

    # ----------------------------------------------------------------------------------
    def encode_many(
        self,
        inputs: Sequence[str],
        targets: Sequence[str],
        batch_size: int = 8,
        pad_to: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a whole dataset; returns ``(token_reps, pooled, mask)`` as numpy.

        Arrays are padded to ``pad_to`` (default: the longest target in the input)
        so that all examples can be compared with plain matrix multiplications.
        """
        torch = self.torch
        was_training = self.lm.training
        self.lm.eval()
        token_batches, pooled_batches, mask_batches = [], [], []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                out = self.forward(inputs[start:start + batch_size], targets[start:start + batch_size])
                token_batches.append(out.token_reps.detach().float().cpu().numpy())
                pooled_batches.append(out.pooled.detach().float().cpu().numpy())
                mask_batches.append(out.mask.detach().cpu().numpy())
        if was_training:
            self.lm.train()
        pad_to = pad_to or max(b.shape[1] for b in token_batches)
        d = token_batches[0].shape[-1]
        token_reps = np.zeros((len(inputs), pad_to, d), dtype=np.float32)
        mask = np.zeros((len(inputs), pad_to), dtype=bool)
        pooled = np.concatenate(pooled_batches, axis=0)
        cursor = 0
        for tb, mb in zip(token_batches, mask_batches):
            n, t = tb.shape[0], tb.shape[1]
            token_reps[cursor:cursor + n, :t] = tb
            mask[cursor:cursor + n, :t] = mb
            cursor += n
        return token_reps, pooled, mask

    # ----------------------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {"mlp": self.mlp.state_dict(), "lm": self.lm.state_dict()}

    def load_state_dict(self, state: Dict) -> None:
        self.mlp.load_state_dict(state["mlp"])
        self.lm.load_state_dict(state["lm"])
