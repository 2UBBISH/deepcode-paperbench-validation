"""Tests for the downstream-task adapters (data pipeline correctness)."""

import torch
from datasets import Dataset, DatasetDict

from data.tasks import (
    QuestionAnsweringTask,
    Seq2SeqClassificationTask,
    SequenceClassificationTask,
    SummarizationTask,
    build_task,
    _f1,
    rouge_scores,
    squad_f1,
)


class _DummyTokenizer:
    """A tiny word-level tokenizer so the tests need no downloads."""

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 3

    def __init__(self):
        self.vocab = {w: i + 10 for i, w in enumerate("abcdefghijklmnopqrstuvwxyz".split())}
        self.vocab.update({"the": 10, "cat": 11, "sat": 12, "mat": 13, "a": 14})
        self.vocab_size = 100

    def __call__(self, text, text_pair=None, max_length=32, padding="max_length",
                 truncation=True, return_offsets_mapping=False, return_overflowing_tokens=False,
                 stride=0, **kwargs):
        single = isinstance(text, str)
        texts = [text] if single else text
        pairs = [None] * len(texts) if text_pair is None else (
            [text_pair] if isinstance(text_pair, str) else text_pair
        )
        out_ids, out_mask, seq_ids, offsets = [], [], [], []
        type_ids = []
        for t, p in zip(texts, pairs):
            ids = [self.cls_token_id]
            seq = [None]
            off = [(0, 0)]
            typ = [0]
            pos = 0
            for w in t.split():
                ids.append(self.vocab.get(w, self.unk_token_id))
                off.append((pos, pos + len(w)))
                seq.append(0)
                typ.append(0)
                pos += len(w) + 1
            if p is not None:
                ids.append(self.sep_token_id)
                seq.append(None)
                off.append((0, 0))
                typ.append(0)
                for w in p.split():
                    ids.append(self.vocab.get(w, self.unk_token_id))
                    off.append((pos, pos + len(w)))
                    seq.append(1)
                    typ.append(1)
                    pos += len(w) + 1
            ids = ids[:max_length]
            off = off[:max_length]
            seq = seq[:max_length]
            typ = typ[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad = max_length - len(ids)
                ids += [self.pad_token_id] * pad
                mask += [0] * pad
                off += [(0, 0)] * pad
                seq += [None] * pad
                typ += [0] * pad
            out_ids.append(ids)
            out_mask.append(mask)
            offsets.append(off)
            seq_ids.append(seq)
            type_ids.append(typ)
        res = {"input_ids": out_ids, "attention_mask": out_mask, "token_type_ids": type_ids}
        if return_offsets_mapping:
            res["offset_mapping"] = offsets
        if return_overflowing_tokens:
            res["overflow_to_sample_mapping"] = list(range(len(texts)))
        res["_seq_ids"] = seq_ids
        return res if not single else {k: v[0] for k, v in res.items()}

    def sequence_ids(self, i):
        return self._last_seq_ids[i]

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(int(i), "") for i in ids if not skip_special_tokens or int(i) > 3).strip()

    def as_target_tokenizer(self):
        import contextlib

        return contextlib.nullcontext()


def test_sequence_classification_tokenization():
    class Tok(_DummyTokenizer):
        def __call__(self, *a, **kw):
            out = super().__call__(*a, **kw)
            self._last_seq_ids = out.pop("_seq_ids")
            return out

    tok = Tok()
    ds = Dataset.from_dict({"sentence": ["a cat", "the mat"], "label": [1, 0]})
    task = SequenceClassificationTask(tok, DatasetDict({"train": ds}), "sst2")
    feats = task.tokenize("train")
    assert "labels" in feats.column_names
    assert feats[0]["input_ids"][0] == tok.cls_token_id
    assert int(feats[0]["labels"]) == 1
    batch = task.collate([feats[0], feats[1]])
    assert batch["input_ids"].shape[0] == 2


def test_squad_span_targets_are_inside_the_context():
    class Tok(_DummyTokenizer):
        def __call__(self, *a, **kw):
            out = super().__call__(*a, **kw)
            self._last_seq_ids = out.pop("_seq_ids")
            return out

    tok = Tok()
    ds = Dataset.from_dict(
        {
            "id": ["q1", "q2"],
            "question": ["cat", "cat"],
            "context": ["the cat sat", "the mat"],
            "answers": [
                {"text": ["cat"], "answer_start": [4]},
                {"text": [], "answer_start": []},
            ],
        }
    )
    task = QuestionAnsweringTask(tok, DatasetDict({"train": ds}))
    feats = task.tokenize("train")
    start, end = int(feats[0]["start_positions"]), int(feats[0]["end_positions"])
    seq = task.tokenizer.sequence_ids(0)
    assert seq[start] == 1 and seq[end] == 1
    # the unanswerable example points at the [CLS] *position* (index 0)
    assert int(feats[1]["start_positions"]) == 0
    assert int(feats[0]["input_ids"][0]) == tok.cls_token_id


def test_squad_f1_metric():
    preds = [{"text": "the cat"}, {"text": ""}]
    refs = [{"answers": {"text": ["the cat"], "answer_start": [0]}}, {"answers": {"text": [], "answer_start": []}}]
    assert squad_f1(preds, refs) == 100.0
    assert _f1("cat sat", "cat sat") == 1.0


def test_rouge_fallback_is_available():
    scores = rouge_scores(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert scores["rouge1"] > 99.0
    assert scores["rougeL"] > 99.0


def test_task_factory_dispatch():
    tok = _DummyTokenizer()
    ds = DatasetDict({"train": Dataset.from_dict({"sentence": ["a"], "label": [0]})})
    assert isinstance(build_task("sst2", ds, tok), SequenceClassificationTask)
    assert isinstance(build_task("sst2", ds, tok, text_to_text=True), Seq2SeqClassificationTask)
