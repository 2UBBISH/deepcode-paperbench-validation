"""Base language model wrapper for the "What Will My Model Forget?" reproduction.

This module provides a thin, model-agnostic wrapper around the seq2seq base PTLMs
used in the paper:

    * ``BART0_L``      -> ``facebook/bart-large`` initialised with INK-USC/ReCross BART0 weights
    * ``FLAN-T5_L``    -> ``google/flan-t5-large``
    * ``FLAN-T5_3B``   -> ``google/flan-t5-3b``
    * ``FLAN-T5_small``-> ``google/flan-t5-small``  (backbone of the encoding function ``h``)

The wrapper exposes exactly the operations required by the rest of the pipeline:

    * ``predict`` / ``generate`` / ``batch_generate`` -- greedy decoding, used to
      compute EM, build ``D_PT_hat`` and collect ``D_R`` (Sec. 2, Sec. 4.1).
    * ``token_logits`` -- teacher-forced per-output-token logits for a (input, target)
      pair, used to build the cached top-k streams ``f0(x_j)``, ``f_i(x_j)`` (Sec. 3.2).
    * ``hidden_states`` / ``encode_token_level`` / ``final_hidden_states`` -- decoder (or
      encoder) final hidden states, i.e. the frozen representation used by the
      "Fixed Logit" baseline and, optionally, by ``h`` (Sec. 3.2, Sec. 4.2).

Note the paper does not specify an optimizer or seeds; the defaults chosen here
(AdamW, seed 42) are documented in ``config/config.yaml``.  Nothing in this module
trains the model: ``src/modeling/refinement.py`` performs the K-step refinement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLM",
    "load_base_lm",
    "MODEL_IDS",
    "DEFAULT_GENERATION_KWARGS",
]

# --------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------
# The primary id is what ``transformers`` loads.  ``weights`` (optional) is a second
# checkpoint that is loaded on top of the architecture for BART0 (ReCross release).
MODEL_IDS: Dict[str, Dict[str, Any]] = {
    "BART0_L": {
        "name": "facebook/bart-large",
        "is_t5": False,
        "n_params": "0.4B",
        "weights": "INK-USC/ReCross",
        "weights_subfolder": "bart0_large",
    },
    "FLAN-T5_L": {"name": "google/flan-t5-large", "is_t5": True, "n_params": "0.8B"},
    "FLAN-T5_3B": {"name": "google/flan-t5-3b", "is_t5": True, "n_params": "3B"},
    "FLAN-T5_small": {"name": "google/flan-t5-small", "is_t5": True, "n_params": "80M"},
}

DEFAULT_GENERATION_KWARGS: Dict[str, Any] = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 64,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}


# --------------------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------------------
class BaseLM(nn.Module):
    """Wrapper around a HF seq2seq model exposing the pipeline's needs.

    Parameters
    ----------
    model:
        A ``transformers`` seq2seq model (already on the desired device/dtype).
    tokenizer:
        The matching tokenizer.
    model_key:
        Registry key, e.g. ``"BART0_L"``.
    max_input_len / max_output_len:
        Truncation lengths for inputs / targets.
    device / dtype:
        Runtime placement and precision.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_key: str = "BART0_L",
        max_input_len: int = 512,
        max_output_len: int = 64,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self._device = torch.device(device)
        self._dtype = dtype
        self.config = getattr(model, "config", None)

    # ------------------------------------------------------------------ properties
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def is_t5(self) -> bool:
        return bool(MODEL_IDS.get(self.model_key, {}).get("is_t5", False))

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.config, "d_model", getattr(self.config, "hidden_size", 768)))

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.config, "vocab_size", 32000))

    @property
    def pad_token_id(self) -> int:
        pid = getattr(self.tokenizer, "pad_token_id", None)
        if pid is None:
            pid = getattr(self.config, "pad_token_id", None)
        if pid is None:
            pid = 0
        return int(pid)

    @property
    def decoder_start_token_id(self) -> int:
        dsid = getattr(self.config, "decoder_start_token_id", None)
        if dsid is None:
            dsid = getattr(self.tokenizer, "bos_token_id", None)
        if dsid is None:
            dsid = getattr(self.tokenizer, "pad_token_id", None)
        if dsid is None:
            dsid = 0
        return int(dsid)

    # ------------------------------------------------------------------ tokenisation
    def tokenize_inputs(
        self,
        inputs: Sequence[str],
        padding: Union[bool, str] = True,
    ) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=self.max_input_len,
        )
        return {k: v.to(self._device) for k, v in enc.items()}

    def tokenize_targets(
        self,
        targets: Optional[Sequence[str]],
        padding: Union[bool, str] = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if targets is None:
            return None
        labels = self.tokenizer(
            list(targets),
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=self.max_output_len,
        )
        return {k: v.to(self._device) for k, v in labels.items()}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        inputs: Union[str, Sequence[str]],
        max_new_tokens: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> List[str]:
        """Greedy-decode one or many inputs and return decoded strings."""
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = list(inputs)
        if len(inputs) == 0:
            return []
        enc = self.tokenize_inputs(inputs)
        kwargs = dict(DEFAULT_GENERATION_KWARGS)
        kwargs["max_new_tokens"] = max_new_tokens or self.max_output_len
        kwargs.update({k: v for k, v in gen_kwargs.items() if v is not None})
        was_training = self.model.training
        self.model.eval()
        out = self.model.generate(**enc, **kwargs)
        if was_training:
            self.model.train()
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def batch_generate(
        self,
        inputs: Sequence[str],
        batch_size: int = 8,
        **gen_kwargs: Any,
    ) -> List[str]:
        """Batched greedy decoding (memory-safe for large models)."""
        inputs = list(inputs)
        preds: List[str] = []
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            preds.extend(self.generate(chunk, **gen_kwargs))
        return preds

    def predict(self, inputs: Union[str, Sequence[str]], **gen_kwargs: Any) -> List[str]:
        """Alias used by ``src/data/dataset_builders.py`` predictor adapters."""
        return self.generate(inputs, **gen_kwargs)

    def predict_batch(
        self, inputs: Sequence[str], batch_size: int = 8, **gen_kwargs: Any
    ) -> List[str]:
        """Alias used by ``src/data/dataset_builders.py`` predictor adapters."""
        return self.batch_generate(inputs, batch_size=batch_size, **gen_kwargs)

    # ------------------------------------------------------------------ logits / h
    @torch.no_grad()
    def token_logits(
        self,
        inputs: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        batch_size: int = 8,
        need_hidden: bool = False,
        use_encoder_hidden: bool = False,
    ) -> Dict[str, Any]:
        """Teacher-forced per-token logits (and optionally hidden states).

        Returns a dict with ``logits`` (``[B, T, V]`` float32 CPU), an optional
        ``hidden`` (``[B, T, d]``) and ``target_ids`` (``[B, T]``).  Teacher forcing
        means the decoded sequence is exactly the reference ``y``; the first target
        token is the decoder start token, hence logits are shifted by one, matching
        the standard seq2seq convention used by the paper's logit definitions.
        """
        inputs = list(inputs)
        if targets is None:
            targets = ["" for _ in inputs]
        targets = list(targets)
        assert len(inputs) == len(targets), "inputs and targets must align"

        all_logits: List[torch.Tensor] = []
        all_hidden: List[torch.Tensor] = []
        all_target_ids: List[torch.Tensor] = []

        was_training = self.model.training
        self.model.eval()
        for start in range(0, len(inputs), batch_size):
            inp_chunk = inputs[start : start + batch_size]
            tgt_chunk = targets[start : start + batch_size]
            enc = self.tokenize_inputs(inp_chunk)
            labels = self.tokenizer(
                tgt_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_output_len,
            )
            label_ids = labels["input_ids"].to(self._device)
            label_mask = labels.get("attention_mask")
            if label_mask is not None:
                label_mask = label_mask.to(self._device)

            model_inputs = dict(enc)
            model_inputs["labels"] = label_ids
            out = self.model(
                **model_inputs,
                output_hidden_states=need_hidden,
                return_dict=True,
            )
            logits = out.logits.float().detach().cpu()  # [B, T, V]
            all_logits.append(logits)
            all_target_ids.append(label_ids.detach().cpu())
            if need_hidden:
                hidden_src = out.encoder_hidden_states if use_encoder_hidden else out.decoder_hidden_states
                hidden = hidden_src[-1].float().detach().cpu()  # [B, T, d]
                all_hidden.append(hidden)
        if was_training:
            self.model.train()

        result: Dict[str, Any] = {
            "logits": torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0),
            "target_ids": torch.cat(all_target_ids, dim=0) if all_target_ids else torch.zeros(0, dtype=torch.long),
        }
        if need_hidden:
            result["hidden"] = torch.cat(all_hidden, dim=0) if all_hidden else torch.zeros(0)
        return result

    @torch.no_grad()
    def hidden_states(
        self,
        inputs: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        batch_size: int = 8,
        use_encoder_hidden: bool = False,
    ) -> torch.Tensor:
        """Final-layer hidden states for (input, target) pairs: ``[B, T, d]``."""
        out = self.token_logits(
            inputs,
            targets,
            batch_size=batch_size,
            need_hidden=True,
            use_encoder_hidden=use_encoder_hidden,
        )
        return out["hidden"]

    def encode_token_level(
        self,
        inputs: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Token-level representation ``h(x, y)`` using frozen model states.

        This is the representation used by the "Fixed Logit" baseline (Sec. 4.2):
        when only the task heads are tuned the transformation is identity, so the
        frozen base-LM final-layer representation is exact.
        """
        return self.hidden_states(inputs, targets, batch_size=batch_size, **kwargs)

    def final_hidden_states(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Alias tolerated by ``src/forecasters/logit_based.py`` auto-detection."""
        return self.encode_token_level(*args, **kwargs)

    # ------------------------------------------------------------------ misc
    def gradient_checkpointing(self, enable: bool = True) -> None:
        try:
            self.model.gradient_checkpointing_enable() if enable else self.model.gradient_checkpointing_disable()
        except Exception as exc:  # pragma: no cover - model dependent
            logger.debug("gradient checkpointing unavailable: %s", exc)

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        out = super().to(*args, **kwargs)
        try:
            dev = kwargs.get("device")
            if dev is not None:
                self._device = torch.device(dev)
        except Exception:
            pass
        return out


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(dtype).lower(), torch.float32)


def load_base_lm(
    model_key: str = "BART0_L",
    device: Union[str, torch.device] = "cuda",
    dtype: Union[str, torch.dtype] = "float32",
    cache_dir: Optional[str] = None,
    max_input_len: int = 512,
    max_output_len: int = 64,
    model_name_or_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    **kwargs: Any,
) -> BaseLM:
    """Load a base PTLM (``f_0``) and wrap it in :class:`BaseLM`.

    ``model_key`` is one of ``BART0_L`` / ``FLAN-T5_L`` / ``FLAN-T5_3B`` / ``FLAN-T5_small``.
    Any extra keyword arguments are forwarded to ``from_pretrained``.
    """
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    spec = MODEL_IDS.get(model_key, {})
    name = model_name_or_path or spec.get("name", model_key)
    torch_dtype = _resolve_dtype(dtype)

    logger.info("Loading base LM %s (%s) on %s as %s", model_key, name, device, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

    # BART0 ships just the state dict of a bart-large; load the architecture then the weights.
    load_kwargs: Dict[str, Any] = dict(cache_dir=cache_dir)
    if torch_dtype in (torch.float16, torch.bfloat16):
        load_kwargs["torch_dtype"] = torch_dtype
    load_kwargs.update(kwargs)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(name, **load_kwargs)
    except Exception:
        # Some FLAN-T5 checkpoints want T5ForConditionalGeneration explicitly.
        from transformers import T5ForConditionalGeneration

        config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(name, config=config, **load_kwargs)

    weights = weights_path or spec.get("weights")
    if weights:
        try:
            _maybe_load_bart0_weights(model, weights, spec.get("weights_subfolder"), cache_dir)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Could not load auxiliary weights '%s': %s", weights, exc)

    model.to(device)
    if torch_dtype in (torch.float16, torch.bfloat16):
        model.to(dtype=torch_dtype)
    model.eval()

    return BaseLM(
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        device=device,
        dtype=torch_dtype,
    )


def _maybe_load_bart0_weights(
    model: nn.Module,
    weights: str,
    subfolder: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Best-effort load of the BART0 (ReCross) fine-tuned weights."""
    from huggingface_hub import hf_hub_download

    for filename in ("pytorch_model.bin", "model.safetensors", "state_dict.pt", "pytorch_model.pt"):
        try:
            path = hf_hub_download(
                repo_id=weights,
                filename=os.path.join(subfolder, filename) if subfolder else filename,
                cache_dir=cache_dir,
            )
        except Exception:
            continue
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if hasattr(model, "load_state_dict"):
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                "Loaded BART0 weights from %s (missing=%d, unexpected=%d)",
                path,
                len(missing),
                len(unexpected),
            )
        return
    raise FileNotFoundError(f"No checkpoint file found in HF repo '{weights}'")


# --------------------------------------------------------------------------------------
# Small CLI for smoke-testing
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="Smoke-test base LM loading/generation.")
    p.add_argument("--model-key", default="FLAN-T5_small", choices=sorted(MODEL_IDS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--prompt", default="Translate English to German: Hello world")
    p.add_argument("--target", default="")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - smoke test
    args = parse_args(argv)
    lm = load_base_lm(args.model_key, device=args.device, dtype=args.dtype)
    print("Generation:", lm.generate([args.prompt])[0])
    if args.target:
        out = lm.token_logits([args.prompt], [args.target])
        print("Logits shape:", tuple(out["logits"].shape))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
