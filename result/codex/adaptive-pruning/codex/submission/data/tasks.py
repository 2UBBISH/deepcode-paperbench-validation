"""Task adapters for GLUE (SST2, MNLI, ...), SQuAD v2.0 and CNN/DailyMail.

Each adapter knows how to

* tokenise the raw HuggingFace dataset (``prepare``),
* run a forward pass that optionally returns the intermediate hidden states
  required by the layer-wise distillation objective (``forward``),
* turn logits into the metric reported by the paper
  (accuracy / F1 / ROUGE-1,2,L).

The paper's main tables use:

* RoBERTa-base: SST2, MNLI (accuracy), SQuAD v2.0 (F1)   -- Table 2
* T5-base:      SST2, MNLI (accuracy), CNN/DM (ROUGE)   -- Table 2
* RoBERTa-base: seven GLUE tasks                        -- Table 8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def squad_f1(preds: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> float:
    """Exact-match + F1 for SQuAD v2.0 (max over the reference answers)."""
    total_f1 = 0.0
    for p, r in zip(preds, refs):
        answers = r.get("answers", {})
        if isinstance(answers, dict):
            golds = answers.get("text", [])
        else:
            golds = answers
        if not golds:
            total_f1 += 1.0 if p["text"] == "" else 0.0
            continue
        best = 0.0
        for g in golds:
            best = max(best, _f1(p["text"], g))
        total_f1 += best
    return 100.0 * total_f1 / max(1, len(preds))


def _normalize(s: str) -> str:
    import re
    import string

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_tokens = _normalize(pred).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = {}
    for t in p_tokens:
        common[t] = min(common.get(t, 0) + 1, g_tokens.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p_tokens)
    recall = same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_scores(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    """ROUGE-1/2/L F-measures.

    Uses ``rouge_score`` when installed, otherwise a self-contained LCS based
    implementation so that the repository has no hard dependency on it.
    """
    try:  # pragma: no cover - depends on the environment
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for p, r in zip(preds, refs):
            sc = scorer.score(r, p)
            for k in agg:
                agg[k] += sc[k].fmeasure
        n = max(1, len(preds))
        return {k: 100.0 * v / n for k, v in agg.items()}
    except Exception:
        r1 = _rouge_n(preds, refs, 1)
        r2 = _rouge_n(preds, refs, 2)
        rl = _rouge_l(preds, refs)
        return {"rouge1": r1, "rouge2": r2, "rougeL": rl}


def _ngrams(tokens: Sequence[str], n: int):
    from collections import Counter

    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(preds, refs, n) -> float:
    total = 0.0
    for p, r in zip(preds, refs):
        pn, rn = _ngrams(p.split(), n), _ngrams(r.split(), n)
        overlap = sum((pn & rn).values())
        if overlap == 0:
            continue
        prec = overlap / max(1, sum(pn.values()))
        rec = overlap / max(1, sum(rn.values()))
        total += 2 * prec * rec / (prec + rec)
    return 100.0 * total / max(1, len(preds))


def _rouge_l(preds, refs) -> float:
    total = 0.0
    for p, r in zip(preds, refs):
        a, b = p.split(), r.split()
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[len(a)][len(b)]
        if lcs == 0:
            continue
        prec = lcs / max(1, len(a))
        rec = lcs / max(1, len(b))
        total += 2 * prec * rec / (prec + rec)
    return 100.0 * total / max(1, len(preds))


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #
class TaskAdapter:
    """Common interface shared by all downstream tasks."""

    name: str = "task"
    metric_name: str = "accuracy"
    num_labels: int = 2
    max_input_length: int = 128
    max_target_length: int = 128

    def __init__(self, tokenizer, dataset, split_eval: str = "validation") -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.split_eval = split_eval

    # -- data ------------------------------------------------------------- #
    def tokenize(self, split: str):
        raise NotImplementedError

    def collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        keys = [k for k in features[0] if k not in ("idx",)]
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [f[k] for f in features]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals)
            else:
                out[k] = torch.tensor(vals, dtype=torch.long)
        if "idx" in features[0]:
            out["idx"] = torch.tensor([f["idx"] for f in features], dtype=torch.long)
        return out

    # -- model ------------------------------------------------------------ #
    def forward(self, model, batch, output_hidden_states: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def metrics(self, logits, batch) -> Dict[str, float]:
        raise NotImplementedError

    def higher_is_better(self) -> bool:
        return True

    def primary_metric(self, metrics: Dict[str, float]) -> float:
        return metrics[self.metric_name]


# --------------------------------------------------------------------------- #
# GLUE
# --------------------------------------------------------------------------- #
GLUE_SPECS = {
    "sst2": dict(
        keys=("sentence", None), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="sst2", epoch_field=None,
    ),
    "mnli": dict(
        keys=("premise", "hypothesis"), num_labels=3, metric="accuracy",
        dataset="nyu-mll/glue", subset="mnli", epoch_field=None,
    ),
    "qqp": dict(
        keys=("question1", "question2"), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="qqp", epoch_field=None,
    ),
    "qnli": dict(
        keys=("question", "sentence"), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="qnli", epoch_field=None,
    ),
    "cola": dict(
        keys=("sentence", None), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="cola", epoch_field=None,
    ),
    "mrpc": dict(
        keys=("sentence1", "sentence2"), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="mrpc", epoch_field=None,
    ),
    "rte": dict(
        keys=("sentence1", "sentence2"), num_labels=2, metric="accuracy",
        dataset="nyu-mll/glue", subset="rte", epoch_field=None,
    ),
    "stsb": dict(
        keys=("sentence1", "sentence2"), num_labels=1, metric="pearson",
        dataset="nyu-mll/glue", subset="stsb", epoch_field=None,
    ),
}


class SequenceClassificationTask(TaskAdapter):
    """GLUE style single-sentence / sentence-pair classification."""

    def __init__(self, tokenizer, dataset, name: str, split_eval: Optional[str] = None) -> None:
        spec = GLUE_SPECS[name]
        super().__init__(tokenizer, dataset, split_eval=split_eval or _default_eval_split(name))
        self.name = name
        self.spec = spec
        self.num_labels = spec["num_labels"]
        self.metric_name = spec["metric"]
        self.max_input_length = 128 if name in {"sst2", "mrpc", "rte", "cola", "stsb"} else 256

    def tokenize(self, split: str):
        keys = self.spec["keys"]

        def _fn(ex):
            first = ex[keys[0]]
            if keys[1] is None:
                return self.tokenizer(
                    first, truncation=True, max_length=self.max_input_length, padding="max_length"
                )
            return self.tokenizer(
                first,
                ex[keys[1]],
                truncation=True,
                max_length=self.max_input_length,
                padding="max_length",
            )

        ds = self.dataset[split]
        ds = ds.map(_fn, batched=False)
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    def forward(self, model, batch, output_hidden_states: bool = False) -> Dict[str, Any]:
        kw = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=output_hidden_states,
        )
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            kw["token_type_ids"] = batch["token_type_ids"]
        out = model(**kw)
        return {
            "loss": out.loss,
            "logits": out.logits,
            "hidden_states": getattr(out, "hidden_states", None),
        }

    def metrics(self, logits, batch) -> Dict[str, float]:
        labels = batch["labels"].detach().cpu().numpy()
        if self.num_labels == 1:
            preds = logits.detach().cpu().numpy().ravel()
            r = np.corrcoef(preds, labels)[0, 1]
            return {"pearson": float(np.nan_to_num(r))}
        preds = logits.detach().cpu().numpy().argmax(-1)
        return {"accuracy": simple_accuracy(preds, labels)}


def _default_eval_split(name: str) -> str:
    if name == "mnli":
        return "validation_matched"
    return "validation"


# --------------------------------------------------------------------------- #
# SQuAD v2.0
# --------------------------------------------------------------------------- #
class QuestionAnsweringTask(TaskAdapter):
    """Extractive QA with the unanswerable handling of SQuAD v2.0."""

    @staticmethod
    def _seq_ids(tok, i):
        """Sequence id (0 = question, 1 = context, None = special) of token ``i``.

        Fast tokenizers expose ``BatchEncoding.sequence_ids``; the fallback uses
        the token type ids so that the span logic also works with a slow
        tokenizer.
        """
        if hasattr(tok, "sequence_ids"):
            try:
                return tok.sequence_ids(i)
            except Exception:  # pragma: no cover - defensive
                pass
        if "token_type_ids" in tok:
            return [int(t) for t in tok["token_type_ids"][i]]
        return None

    def __init__(self, tokenizer, dataset, split_eval: str = "validation") -> None:
        super().__init__(tokenizer, dataset, split_eval=split_eval)
        self.name = "squad"
        self.metric_name = "f1"
        self.max_input_length = 384
        self.doc_stride = 128
        self.max_answer_length = 30

    def _prepare_train(self, examples):
        """Tokenise a *batch* of SQuAD examples into (start, end) span targets.

        Impossible (unanswerable) questions get the ``[CLS]`` index as their
        target, which is exactly how SQuAD v2.0 scores the "no answer" class.
        """
        questions = [q.strip() for q in examples["question"]]
        tok = self.tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=self.max_input_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        start_positions, end_positions = [], []
        for i, offsets in enumerate(tok["offset_mapping"]):
            seq_ids = self._seq_ids(tok, i)
            if seq_ids is None:
                seq_ids = [1] * len(tok["input_ids"][i])
            cls_idx = tok["input_ids"][i].index(self.tokenizer.cls_token_id)
            example = examples["answers"][sample_map[i]]
            if len(example["answer_start"]) == 0:
                start_positions.append(cls_idx)
                end_positions.append(cls_idx)
                continue

            start_char = example["answer_start"][0]
            end_char = start_char + len(example["text"][0])
            token_start = 0
            while seq_ids[token_start] != 1:
                token_start += 1
            token_end = len(tok["input_ids"][i]) - 1
            while seq_ids[token_end] != 1:
                token_end -= 1

            if offsets[token_start][0] > start_char or offsets[token_end][1] < end_char:
                # the answer was truncated away -> treat as unanswerable
                start_positions.append(cls_idx)
                end_positions.append(cls_idx)
                continue
            while token_start <= token_end and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)
            while token_end >= token_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

        tok["start_positions"] = start_positions
        tok["end_positions"] = end_positions
        tok["example_id"] = [examples["id"][sample_map[i]] for i in range(len(start_positions))]
        tok.pop("offset_mapping")
        return tok

    def tokenize(self, split: str):
        if split == "train":
            ds = self.dataset[split].map(
                self._prepare_train,
                batched=True,
                remove_columns=self.dataset[split].column_names,
            )
            ds = ds.remove_columns(
                [c for c in ds.column_names if c not in {"input_ids", "attention_mask", "start_positions", "end_positions", "token_type_ids"}]
            )
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
            return ds

        def _prep(examples):
            questions = [q.strip() for q in examples["question"]]
            tok = self.tokenizer(
                questions,
                examples["context"],
                truncation="only_second",
                max_length=self.max_input_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            sample_map = tok.pop("overflow_to_sample_mapping")
            tok["example_id"] = [examples["id"][i] for i in sample_map]
            seq_all = [self._seq_ids(tok, i) for i in range(len(tok["input_ids"]))]
            tok["offset_mapping"] = [
                [(o if s == 1 else None) for o, s in zip(offs, seq_all[i])]
                for i, offs in enumerate(tok["offset_mapping"])
            ]
            return tok

        ds = self.dataset[split].map(
            _prep, batched=True, remove_columns=self.dataset[split].column_names
        )
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return ds

    def forward(self, model, batch, output_hidden_states: bool = False) -> Dict[str, Any]:
        kw = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=output_hidden_states,
        )
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            kw["token_type_ids"] = batch["token_type_ids"]
        if "start_positions" in batch:
            kw["start_positions"] = batch["start_positions"]
            kw["end_positions"] = batch["end_positions"]
        out = model(**kw)
        if "start_positions" in batch:
            loss = out.loss
        else:
            start_logits, end_logits = out.start_logits, out.end_logits
            loss = (
                F.cross_entropy(start_logits, start_logits.new_zeros(start_logits.size(0), dtype=torch.long))
                + F.cross_entropy(end_logits, end_logits.new_zeros(end_logits.size(0), dtype=torch.long))
            ) / 2.0
        return {
            "loss": loss,
            "logits": (out.start_logits, out.end_logits),
            "hidden_states": getattr(out, "hidden_states", None),
        }

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, model, features, raw_dataset, batch_size: int = 32, device="cpu") -> Dict[str, float]:
        model.eval()
        starts, ends = [], []
        n = len(features)
        for i in range(0, n, batch_size):
            chunk = [features[j] for j in range(i, min(i + batch_size, n))]
            batch = self.collate(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self.forward(model, batch, output_hidden_states=False)
            s, e = out["logits"]
            starts.append(s.cpu().numpy())
            ends.append(e.cpu().numpy())
        starts = np.concatenate(starts, 0) if starts else np.zeros((0, 1))
        ends = np.concatenate(ends, 0) if ends else np.zeros((0, 1))

        example_to_features: Dict[str, List[int]] = {}
        for fi, f in enumerate(features):
            example_to_features.setdefault(f["example_id"], []).append(fi)

        preds, refs = [], []
        for ex in raw_dataset:
            subs = example_to_features.get(ex["id"], [])
            best = (0.0, "")
            for fi in subs:
                start_logits = starts[fi]
                end_logits = ends[fi]
                input_ids = np.array(features[fi]["input_ids"])
                offsets = features[fi]["offset_mapping"]
                for si in np.argsort(-start_logits)[:20]:
                    for ei in np.argsort(-end_logits)[:20]:
                        if si >= len(offsets) or ei >= len(offsets):
                            continue
                        if ei < si or offsets[si] is None or offsets[ei] is None:
                            continue
                        if ei - si + 1 > self.max_answer_length:
                            continue
                        score = float(start_logits[si] + end_logits[ei])
                        if score > best[0]:
                            text = self.tokenizer.decode(
                                input_ids[si : ei + 1], skip_special_tokens=True
                            )
                            best = (score, text)
            if best[0] < 0.0:
                best = (best[0], "")
            preds.append({"text": best[1]})
            refs.append({"answers": ex["answers"]})
        return {"f1": squad_f1(preds, refs)}


# --------------------------------------------------------------------------- #
# CNN/DailyMail summarisation
# --------------------------------------------------------------------------- #
class SummarizationTask(TaskAdapter):
    """Abstractive summarisation with a T5 encoder-decoder."""

    def __init__(self, tokenizer, dataset, split_eval: str = "validation") -> None:
        super().__init__(tokenizer, dataset, split_eval=split_eval)
        self.name = "cnn_dm"
        self.metric_name = "rouge1"
        self.max_input_length = 512
        self.max_target_length = 128

    def tokenize(self, split: str):
        prefix = "summarize: "
        ds = self.dataset[split]
        article_field = "article" if "article" in ds.column_names else "text"
        summary_field = "highlights" if "highlights" in ds.column_names else "summary"

        def _fn(ex):
            model_inputs = self.tokenizer(
                [prefix + a for a in ex[article_field]],
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    ex[summary_field],
                    max_length=self.max_target_length,
                    padding="max_length",
                    truncation=True,
                )
            model_inputs["labels"] = [
                [(t if t != self.tokenizer.pad_token_id else -100) for t in ids]
                for ids in labels["input_ids"]
            ]
            return model_inputs

        ds = ds.map(_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    def forward(self, model, batch, output_hidden_states: bool = False) -> Dict[str, Any]:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=output_hidden_states,
        )
        return {
            "loss": out.loss,
            "logits": out.logits,
            "hidden_states": (getattr(out, "encoder_hidden_states", None), getattr(out, "decoder_hidden_states", None)),
        }

    def metrics(self, logits, batch) -> Dict[str, float]:
        return {}

    @torch.no_grad()
    def evaluate(self, model, features, raw_dataset, batch_size: int = 16, device="cpu") -> Dict[str, float]:
        model.eval()
        preds, refs = [], []
        decode_batch = 16
        for i in range(0, len(features), batch_size):
            chunk = [features[j] for j in range(i, min(i + batch_size, len(features)))]
            batch = self.collate(chunk)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            preds += self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        for ex in raw_dataset:
            key = "highlights" if "highlights" in ex else "summary"
            refs.append(ex[key])
        return rouge_scores(preds, refs)


# --------------------------------------------------------------------------- #
# Text-to-text classification (T5 on GLUE)
# --------------------------------------------------------------------------- #
T5_VERBALIZERS = {
    "sst2": ["negative", "positive"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "qqp": ["no", "yes"],
    "qnli": ["yes", "no"],
    "rte": ["entailment", "not_entailment"],
    "cola": ["unacceptable", "acceptable"],
}

T5_TEMPLATES = {
    "sst2": "{s0}",
    "mnli": "mnli premise: {s0} hypothesis: {s1}",
    "qqp": "qqp question1: {s0} question2: {s1}",
    "qnli": "qnli question: {s0} sentence: {s1}",
    "rte": "rte sentence1: {s0} sentence2: {s1}",
    "cola": "cola sentence: {s0}",
}


class Seq2SeqClassificationTask(TaskAdapter):
    """T5-style text-to-text formulation of a GLUE classification task."""

    def __init__(self, tokenizer, dataset, name: str, split_eval: Optional[str] = None) -> None:
        super().__init__(tokenizer, dataset, split_eval=split_eval or _default_eval_split(name))
        self.name = name
        self.spec = GLUE_SPECS[name]
        self.verbalizers = T5_VERBALIZERS[name]
        self.template = T5_TEMPLATES[name]
        self.num_labels = self.spec["num_labels"]
        self.metric_name = self.spec["metric"]
        self.max_input_length = 128
        self.max_target_length = 8

    def _text(self, ex) -> str:
        keys = self.spec["keys"]
        subs = {"s0": ex[keys[0]]}
        if keys[1] is not None:
            subs["s1"] = ex[keys[1]]
        return self.template.format(**subs)

    def tokenize(self, split: str):
        ds = self.dataset[split]
        tok = self.tokenizer

        def _fn(ex):
            enc = tok(self._text(ex), truncation=True, max_length=self.max_input_length, padding="max_length")
            target = self.verbalizers[ex["label"]]
            with tok.as_target_tokenizer():
                labels = tok(target, truncation=True, max_length=self.max_target_length, padding="max_length")
            enc["labels"] = [
                (t if t != tok.pad_token_id else -100) for t in labels["input_ids"]
            ]
            return enc

        ds = ds.map(_fn, batched=False, remove_columns=ds.column_names)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    def forward(self, model, batch, output_hidden_states: bool = False) -> Dict[str, Any]:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=output_hidden_states,
        )
        return {
            "loss": out.loss,
            "logits": out.logits,
            "hidden_states": (
                getattr(out, "encoder_hidden_states", None),
                getattr(out, "decoder_hidden_states", None),
            ),
        }

    def metrics(self, logits, batch) -> Dict[str, float]:
        preds = logits.argmax(-1).detach().cpu().numpy()
        return {"accuracy": float((preds == 0).mean())}

    @torch.no_grad()
    def evaluate(self, model, features, raw_dataset, batch_size: int = 128, device="cpu") -> Dict[str, float]:
        model.eval()
        correct, total = 0, 0
        for i in range(0, len(features), batch_size):
            chunk = [features[j] for j in range(i, min(i + batch_size, len(features)))]
            batch = self.collate(chunk)
            generated = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=self.max_target_length,
                num_beams=1,
            )
            texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for t, ex in zip(texts, raw_dataset[i : i + len(chunk)]):
                gold = self.verbalizers[ex["label"]]
                correct += int(t.strip() == gold)
                total += 1
        return {"accuracy": 100.0 * correct / max(1, total)}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_task(name: str, dataset, tokenizer, text_to_text: bool = False) -> TaskAdapter:
    """Instantiate the adapter for ``name`` (one of the paper's datasets)."""
    key = name.lower()
    if key in GLUE_SPECS:
        if text_to_text:
            return Seq2SeqClassificationTask(tokenizer, dataset, key)
        return SequenceClassificationTask(tokenizer, dataset, key)
    if key in {"squad", "squad_v2", "squadv2"}:
        return QuestionAnsweringTask(tokenizer, dataset)
    if key in {"cnn_dm", "cnndm", "cnn_dailymail", "cnn/dm"}:
        return SummarizationTask(tokenizer, dataset)
    raise ValueError(f"unknown task '{name}'")


__all__ = [
    "SequenceClassificationTask",
    "Seq2SeqClassificationTask",
    "QuestionAnsweringTask",
    "SummarizationTask",
    "build_task",
    "squad_f1",
    "rouge_scores",
]
