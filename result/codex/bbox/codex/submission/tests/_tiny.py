"""Helpers building a tiny, fully offline adapter for the unit tests."""

from __future__ import annotations

import os
from typing import List, Optional

import torch


DEFAULT_VOCAB: List[str] = [
    "what", "is", "the", "answer", "to", "of", "and", "+", "-", "*", "=", "?",
    "q", "a", "step", "one", "two", "three", "target", "correct", "wrong",
    "final", "partial", "reasoning", "done", "x", "four", "five", "six",
    "eight", "ten", "twelve", "18", "19", "2", "3", "4", "5", "6", "7", "8",
    "9", "10", "11", "12", "15", "20", "adding", "numbers", "gives", "the",
    "sky", "blue", "green", "yes", "no", "it", "because", "passes", "through",
    "watermelon", "seeds", "during", "day", "night", "longer", "reasoning.",
]


def build_tiny_tokenizer(vocab: Optional[List[str]] = None):
    """A whitespace ``WordLevel`` tokenizer that needs no download."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(
        vocab if vocab is not None else DEFAULT_VOCAB
    )
    vocab_dict = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    backend = Tokenizer(WordLevel(vocab_dict, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def build_tiny_encoder(vocab_size: int, hidden_size: int = 32, num_layers: int = 2):
    from transformers import DebertaV2Config, DebertaV2Model

    config = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=128,
        type_vocab_size=0,
    )
    return DebertaV2Model(config)


def build_tiny_adapter(extra_vocab: Optional[List[str]] = None):
    from bbox_adapter.adapter.energy import EnergyAdapter

    tokenizer = build_tiny_tokenizer(extra_vocab)
    encoder = build_tiny_encoder(vocab_size=len(tokenizer))
    return EnergyAdapter(
        encoder=encoder,
        tokenizer=tokenizer,
        name="tiny-deberta",
        max_length=128,
        device="cpu",
    )


def save_tiny_adapter(directory: str) -> str:
    """Persist a tiny adapter so that ``build_adapter`` can load it by path."""

    adapter = build_tiny_adapter()
    adapter.save(directory)
    return directory


def save_tiny_encoder(directory: str) -> str:
    """Persist a tiny encoder + tokenizer the way ``AutoModel`` expects it."""

    tokenizer = build_tiny_tokenizer()
    encoder = build_tiny_encoder(vocab_size=len(tokenizer))
    os.makedirs(directory, exist_ok=True)
    encoder.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
