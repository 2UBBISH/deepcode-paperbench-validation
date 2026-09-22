"""CNN/DailyMail summarization pipeline for APT.

Implements the CNN/DM data pipeline used in APT (paper Sec. 5.1 / Appendix A,
Table 6):

    "For T5 models, we also fine-tune them on CNN/DM (Nallapati et al., 2016)
     and report the ROUGE 1/2/L scores."

Table 6 hyper-parameters for CNN/DM are:

    ==================  ============
    Learning rate       1e-4
    Batch size          16
    Epochs              16
    Distill epochs      6
    ==================  ============

The model is ``t5-lm-adapt`` (T5 trained only on C4), so the task is framed as
text-to-text with the ``"summarize: "`` prefix and targets truncated to 128
tokens, which is the standard T5 summarization setup used by the paper's
baselines (FT / LoRA / LoRA+Prune / APT).

This module is intentionally dependency-light: heavy packages (``datasets``,
``torch``, ``rouge_score``) are imported lazily so the module and its
``_self_test`` work in a bare environment.  When ``rouge_score`` is missing we
fall back to a self-contained ROUGE implementation that mirrors the official
count-based formulation (clipped n-gram matches, sentence-level LCS), which
keeps the reported ROUGE-1/2/L numbers comparable.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Defaults (Table 6 + standard T5 summarization setup)
# ---------------------------------------------------------------------------

CNN_DM_HF_NAME = "cnn_dailymail"
CNN_DM_HF_CONFIG = "3.0.0"
CNN_DM_TASK_PREFIX = "summarize: "

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
MIN_TARGET_LENGTH = 0

# Generation defaults used for ROUGE evaluation (standard T5 summarization).
GEN_NUM_BEAMS = 4
GEN_LENGTH_PENALTY = 2.0
GEN_MAX_LENGTH = 128
GEN_MIN_LENGTH = 0
GEN_NO_REPEAT_NGRAM_SIZE = 3

IGNORE_INDEX = -100

CNN_DM_TABLE6: Dict[str, Any] = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 16,
    "distill_epochs": 6,
    "target_sparsity": 0.60,
    "max_source_length": MAX_SOURCE_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    # Distillation loss weighting for CNN/DM (paper Sec. 4.4 / Addendum):
    # 0.1 * L_pred + 0.9 * L_layer
    "pred_distill_weight": 0.1,
    "layer_distill_weight": 0.9,
}

MULTIREF_SEPARATOR = " ||| "  # CNN/DM multi-reference separator in HF ``datasets``


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------


@dataclass
class CnndmSpec:
    """Static description of the CNN/DM summarization task."""

    name: str = "cnndm"
    hf_name: str = CNN_DM_HF_NAME
    hf_config: str = CNN_DM_HF_CONFIG
    source_key: str = "article"
    target_key: str = "highlights"
    task_prefix: str = CNN_DM_TASK_PREFIX
    metric: str = "rouge"
    max_source_length: int = MAX_SOURCE_LENGTH
    max_target_length: int = MAX_TARGET_LENGTH

    @property
    def input_prefix(self) -> str:
        return self.task_prefix


CNNDM_SPEC = CnndmSpec()


# ---------------------------------------------------------------------------
# Raw data loading (same conventions as ``apt/data/glue.py`` / ``squad.py``)
# ---------------------------------------------------------------------------


def canonical_split(split: str) -> str:
    """Map split aliases to CNN/DM split names."""
    key = str(split).strip().lower()
    mapping = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
        "development": "validation",
        "test": "test",
    }
    if key not in mapping:
        raise ValueError(
            f"Unknown CNN/DM split {split!r}; expected one of "
            f"train/validation/test"
        )
    return mapping[key]


def load_cnndm_split(
    split: str = "validation",
    *,
    data_dir: Optional[str] = None,
    hf_name: str = CNN_DM_HF_NAME,
    hf_config: str = CNN_DM_HF_CONFIG,
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load raw CNN/DM records.

    Records are dictionaries with (at least) ``article`` and ``highlights``
    keys.  Uses HuggingFace ``datasets`` when available; otherwise falls back to
    offline JSON/JSONL files named ``{hf_name}.{split}.jsonl`` (or ``.json``)
    under ``data_dir``.
    """
    split = canonical_split(split)
    records: Optional[List[Dict[str, Any]]] = None

    if use_datasets:
        try:  # pragma: no cover - network dependent
            from datasets import load_dataset  # type: ignore

            try:
                ds = load_dataset(
                    hf_name,
                    hf_config,
                    split=split,
                    cache_dir=cache_dir,
                )
            except Exception:
                ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)
            records = [dict(x) for x in ds]
        except Exception:
            records = None

    if records is None:
        records = _load_offline_cnndm(split, data_dir=data_dir, hf_name=hf_name)

    if max_examples is not None:
        records = records[: int(max_examples)]
    return records


def _load_offline_cnndm(
    split: str,
    *,
    data_dir: Optional[str] = None,
    hf_name: str = CNN_DM_HF_NAME,
) -> List[Dict[str, Any]]:
    if not data_dir:
        raise ValueError(
            "Could not load CNN/DM via `datasets` and no `data_dir` was given "
            "for the offline fallback. Expected "
            f"{hf_name}.{split}.jsonl or .json inside `data_dir`."
        )
    candidates = [
        os.path.join(data_dir, f"{hf_name}.{split}.jsonl"),
        os.path.join(data_dir, f"{hf_name}.{split}.json"),
        os.path.join(data_dir, f"{hf_name}.{hf_config}.{split}.jsonl"),
        os.path.join(data_dir, f"{hf_name}.{hf_config}.{split}.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in handle if line.strip()]
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "examples", "articles"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported CNN/DM file layout in {path}")
        return [dict(x) for x in payload]
    raise FileNotFoundError(
        f"No offline CNN/DM file found for split {split!r} in {data_dir!r}"
    )


def examples_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    spec: CnndmSpec = CNNDM_SPEC,
) -> List[Dict[str, Any]]:
    """Normalize raw records into ``{"document", "summary", "id"}`` dicts."""
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        document = (
            record.get(spec.source_key)
            or record.get("document")
            or record.get("text")
            or ""
        )
        summary = record.get(spec.target_key) or record.get("summary") or ""
        examples.append(
            {
                "id": str(record.get("id", idx)),
                "document": _clean_text(document),
                "summary": _clean_text(summary),
            }
        )
    return examples


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = MULTIREF_SEPARATOR.join(str(t) for t in text)
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


@dataclass
class CnndmDataset:
    """Tokenized CNN/DM examples for a seq2seq (T5) model."""

    examples: List[Dict[str, Any]]
    tokenizer: Any
    spec: CnndmSpec = CNNDM_SPEC
    max_source_length: int = MAX_SOURCE_LENGTH
    max_target_length: int = MAX_TARGET_LENGTH
    add_prefix: bool = True
    preprocess_summary: bool = True

    def __post_init__(self) -> None:
        self._features: List[Dict[str, Any]] = [
            self._tokenize(example) for example in self.examples
        ]

    def _source_text(self, example: Dict[str, Any]) -> str:
        document = example.get("document", "")
        if self.add_prefix and self.spec.task_prefix:
            return f"{self.spec.task_prefix}{document}"
        return document

    def _target_text(self, example: Dict[str, Any]) -> str:
        summary = example.get("summary", "")
        if self.preprocess_summary and summary:
            summary = summary
        return summary

    def _tokenize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        source = self._source_text(example)
        target = self._target_text(example)

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        with_targets = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        labels = list(with_targets["input_ids"])

        feature: Dict[str, Any] = {
            "input_ids": list(model_inputs["input_ids"]),
            "attention_mask": list(model_inputs.get("attention_mask", [])),
            "labels": labels,
            "id": example.get("id", ""),
            "summary": target,
        }
        if "token_type_ids" in model_inputs:
            feature["token_type_ids"] = list(model_inputs["token_type_ids"])
        return feature

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._features[index]

    @property
    def features(self) -> List[Dict[str, Any]]:
        return self._features


class CnndmCollator:
    """Pads variable-length CNN/DM batches to the longest item in the batch."""

    def __init__(
        self,
        tokenizer: Any = None,
        pad_to_multiple_of: Optional[int] = 8,
        label_pad_token_id: int = IGNORE_INDEX,
        include_token_type_ids: bool = False,
        return_tensors: str = "pt",
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.include_token_type_ids = include_token_type_ids
        self.return_tensors = return_tensors

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            return int(self.tokenizer.pad_token_id)
        return 0

    def _pad(self, sequences: List[List[int]], value: int) -> Any:
        import torch  # lazy

        max_len = max((len(seq) for seq in sequences), default=0)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len += self.pad_to_multiple_of - (max_len % self.pad_to_multiple_of)
        max_len = max(max_len, 1)
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad"):
            padded = self.tokenizer.pad(
                {"input_ids": sequences},
                padding="max_length",
                max_length=max_len,
                return_tensors=self.return_tensors,
            )
            return padded["input_ids"]
        out = torch.full((len(sequences), max_len), int(value), dtype=torch.long)
        for row, seq in enumerate(sequences):
            if seq:
                out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch  # lazy

        features = list(features)
        input_ids = self._pad([f["input_ids"] for f in features], self.pad_token_id)
        attention_mask = self._pad(
            [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features], 0
        )

        labels = self._pad_labels([f["labels"] for f in features])

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if self.include_token_type_ids and all("token_type_ids" in f for f in features):
            batch["token_type_ids"] = self._pad(
                [f["token_type_ids"] for f in features], 0
            )
        if any("id" in f for f in features):
            batch["id"] = [f.get("id", "") for f in features]
        if any("summary" in f for f in features):
            batch["summary"] = [f.get("summary", "") for f in features]
        return batch

    def _pad_labels(self, sequences: List[List[int]]) -> Any:
        import torch  # lazy

        max_len = max((len(seq) for seq in sequences), default=0)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len += self.pad_to_multiple_of - (max_len % self.pad_to_multiple_of)
        max_len = max(max_len, 1)
        out = torch.full(
            (len(sequences), max_len), int(self.label_pad_token_id), dtype=torch.long
        )
        for row, seq in enumerate(sequences):
            if seq:
                out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out


def build_cnndm_dataset(
    tokenizer: Any,
    split: str = "validation",
    *,
    max_source_length: int = MAX_SOURCE_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
    data_dir: Optional[str] = None,
    hf_name: str = CNN_DM_HF_NAME,
    hf_config: str = CNN_DM_HF_CONFIG,
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
    add_prefix: bool = True,
    max_examples: Optional[int] = None,
    records: Optional[Iterable[Dict[str, Any]]] = None,
    spec: CnndmSpec = CNNDM_SPEC,
) -> CnndmDataset:
    """Load + tokenize one CNN/DM split."""
    if records is None:
        records = load_cnndm_split(
            split,
            data_dir=data_dir,
            hf_name=hf_name,
            hf_config=hf_config,
            use_datasets=use_datasets,
            cache_dir=cache_dir,
            max_examples=max_examples,
        )
    examples = examples_from_records(records, spec=spec)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    return CnndmDataset(
        examples=examples,
        tokenizer=tokenizer,
        spec=spec,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_prefix=add_prefix,
    )


def make_cnndm_dataloaders(
    tokenizer: Any,
    splits: Sequence[str] = ("train", "validation"),
    *,
    batch_size: Optional[int] = None,
    max_source_length: int = MAX_SOURCE_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    data_dir: Optional[str] = None,
    hf_name: str = CNN_DM_HF_NAME,
    hf_config: str = CNN_DM_HF_CONFIG,
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = 8,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build ``{split: DataLoader}`` for CNN/DM using Table 6 defaults."""
    import torch  # lazy
    from torch.utils.data import DataLoader  # lazy

    if batch_size is None:
        batch_size = CNN_DM_TABLE6["batch_size"]

    collator = CnndmCollator(
        tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of
    )

    loaders: Dict[str, Any] = {}
    for split in splits:
        dataset = build_cnndm_dataset(
            tokenizer,
            split,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            data_dir=data_dir,
            hf_name=hf_name,
            hf_config=hf_config,
            use_datasets=use_datasets,
            cache_dir=cache_dir,
            max_examples=max_examples,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[canonical_split(split)] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle_train and canonical_split(split) == "train"),
            num_workers=num_workers,
            collate_fn=collator,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def get_cnndm_hparams(table: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the CNN/DM Table 6 hyper-parameters."""
    if table is not None and "cnndm" in table:
        entry = table["cnndm"]
        return dict(entry) if isinstance(entry, dict) else vars(entry)
    try:  # pragma: no cover - optional cross-module consistency
        from apt.data.glue import TABLE6  # type: ignore

        if "cnndm" in TABLE6:
            entry = TABLE6["cnndm"]
            return dict(entry) if isinstance(entry, dict) else vars(entry)
    except Exception:
        pass
    return dict(CNN_DM_TABLE6)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_summaries(
    model: Any,
    tokenizer: Any,
    dataloader: Any,
    *,
    max_length: int = GEN_MAX_LENGTH,
    min_length: int = GEN_MIN_LENGTH,
    num_beams: int = GEN_NUM_BEAMS,
    length_penalty: float = GEN_LENGTH_PENALTY,
    no_repeat_ngram_size: int = GEN_NO_REPEAT_NGRAM_SIZE,
    device: Any = None,
    max_batches: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """Greedy/beam generation over a CNN/DM dataloader.

    Returns ``(predictions, references)`` as lists of decoded strings.  When the
    batch carries the raw ``summary`` field it is used as the reference;
    otherwise labels are decoded (``-100`` positions removed).
    """
    import torch  # lazy

    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - degenerate model
            device = torch.device("cpu")

    pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    predictions: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=num_beams > 1,
                pad_token_id=pad_token_id,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(_clean_text(text) for text in decoded)

            if "summary" in batch:
                references.extend(_clean_text(text) for text in batch["summary"])
            else:
                labels = batch["labels"].clone()
                labels[labels == IGNORE_INDEX] = pad_token_id
                decoded_refs = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                references.extend(_clean_text(text) for text in decoded_refs)

            if verbose and step % 50 == 0:
                print(f"[cnndm] generated {step} batches")

    return predictions, references


def decode_predictions(token_ids: Any, tokenizer: Any) -> List[str]:
    """Decode generated token ids to cleaned strings."""
    decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return [_clean_text(text) for text in decoded]


# ---------------------------------------------------------------------------
# ROUGE metrics
# ---------------------------------------------------------------------------


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[Any],
    *,
    use_stemmer: bool = True,
    aggregate: str = "mean",
) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores (x100) as reported in the paper.

    ``references`` may be a list of strings or a list of lists of strings
    (CNN/DM has multiple reference summaries).  Uses the official
    ``rouge_score`` package when installed, otherwise a self-contained
    fallback with the same clipped-count formulation.
    """
    predictions = [_clean_text(p) for p in predictions]
    references = [_normalize_references(r) for r in references]
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:  # pragma: no cover - depends on optional dependency
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
        )
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, refs in zip(predictions, references):
            best = None
            for ref in refs:
                scores = scorer.score(ref, pred)
                fmeasures = {
                    key: float(value.fmeasure) for key, value in scores.items()
                }
                if best is None or fmeasures["rouge1"] > best["rouge1"]:
                    best = fmeasures
            if best is None:
                best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            for key in totals:
                totals[key] += best[key]
        n = len(predictions)
        return {key: 100.0 * value / n for key, value in totals.items()}
    except Exception:
        pass

    # Self-contained fallback (clipped n-gram overlap + sentence-level LCS).
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, refs in zip(predictions, references):
        pred_tokens = _tokenize_for_rouge(pred)
        best = None
        for ref in refs:
            ref_tokens = _tokenize_for_rouge(ref)
            candidate = {
                "rouge1": _rouge_n_f1(pred_tokens, ref_tokens, 1),
                "rouge2": _rouge_n_f1(pred_tokens, ref_tokens, 2),
                "rougeL": _rouge_l_f1(pred_tokens, ref_tokens),
            }
            if best is None or candidate["rouge1"] > best["rouge1"]:
                best = candidate
        if best is None:
            best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for key in totals:
            totals[key] += best[key]

    n = len(predictions)
    if aggregate == "sum":
        return {key: value for key, value in totals.items()}
    return {key: 100.0 * value / n for key, value in totals.items()}


def _normalize_references(references: Any) -> List[List[str]]:
    refs: List[List[str]] = []
    for ref in references:
        if isinstance(ref, str):
            refs.append([_clean_text(r) for r in ref.split(MULTIREF_SEPARATOR)])
        elif isinstance(ref, (list, tuple)):
            refs.append([_clean_text(r) for r in ref])
        else:
            refs.append([_clean_text(str(ref))])
    return refs


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_for_rouge(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(pred: Sequence[str], ref: Sequence[str], n: int) -> float:
    if len(pred) < n or len(ref) < n:
        return 0.0
    pred_counts = _ngrams(pred, n)
    ref_counts = _ngrams(ref, n)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    return _f1(precision, recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def _rouge_l_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    return _f1(lcs / len(pred), lcs / len(ref))


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_cnndm_metrics(
    predictions: Sequence[str],
    references: Sequence[Any],
    *,
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """Alias of :func:`compute_rouge` returning ROUGE-1/2/L F1."""
    return compute_rouge(predictions, references, use_stemmer=use_stemmer)


def rouge_metric_summary(metrics: Dict[str, float]) -> str:
    """``R1/R2/RL`` human readable summary in the paper's ordering."""
    return (
        f"ROUGE-1: {metrics.get('rouge1', 0.0):.1f}  "
        f"ROUGE-2: {metrics.get('rouge2', 0.0):.1f}  "
        f"ROUGE-L: {metrics.get('rougeL', 0.0):.1f}"
    )


def evaluate_cnndm(
    model: Any,
    tokenizer: Any,
    dataloader: Any,
    *,
    use_stemmer: bool = True,
    max_batches: Optional[int] = None,
    **generate_kwargs: Any,
) -> Dict[str, float]:
    """Generate summaries and compute ROUGE-1/2/L in one call."""
    predictions, references = generate_summaries(
        model,
        tokenizer,
        dataloader,
        max_batches=max_batches,
        **generate_kwargs,
    )
    return compute_rouge(predictions, references, use_stemmer=use_stemmer)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Dependency-light sanity checks (no network / no torch required)."""
    # 1) Table 6 hyper-parameters for CNN/DM.
    hparams = get_cnndm_hparams()
    assert hparams["learning_rate"] == 1e-4, hparams
    assert hparams["batch_size"] == 16, hparams
    assert hparams["epochs"] == 16, hparams
    assert hparams["distill_epochs"] == 6, hparams
    assert hparams["target_sparsity"] == 0.6, hparams
    assert hparams["pred_distill_weight"] == 0.1, hparams
    assert hparams["layer_distill_weight"] == 0.9, hparams

    # 2) Split aliasing.
    assert canonical_split("dev") == "validation"
    assert canonical_split("val") == "validation"
    assert canonical_split("test") == "test"

    # 3) Record normalization + text cleaning (newlines collapsed).
    records = [
        {
            "article": "Line one.\nLine two, with   spaces.",
            "highlights": "Line one.\nLine two.",
            "id": "abc",
        }
    ]
    examples = examples_from_records(records)
    assert examples[0]["document"] == "Line one. Line two, with spaces."
    assert examples[0]["summary"] == "Line one. Line two."
    assert examples[0]["id"] == "abc"

    # 4) ROUGE fallback: identical text -> perfect scores, disjoint -> 0.
    perfect = compute_rouge(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert abs(perfect["rouge1"] - 100.0) < 1e-6, perfect
    assert abs(perfect["rouge2"] - 100.0) < 1e-6, perfect
    assert abs(perfect["rougeL"] - 100.0) < 1e-6, perfect

    zero = compute_rouge(["alpha beta"], ["gamma delta"])
    assert zero["rouge1"] == 0.0 and zero["rouge2"] == 0.0 and zero["rougeL"] == 0.0

    # Partial overlap: pred shares 2 of 3 unigrams with the reference.
    partial = compute_rouge(["a b c"], ["a b"])
    assert abs(partial["rouge1"] - 80.0) < 1e-6, partial  # 2/3 & 2/2 -> F1 0.8

    # 5) Multi-reference handling: best reference is chosen.
    multi = compute_rouge(["the cat"], [["a dog", "the cat"]])
    assert abs(multi["rouge1"] - 100.0) < 1e-6, multi
    assert abs(multi["rougeL"] - 100.0) < 1e-6, multi

    # 6) ROUGE-L uses longest-common-subsequence ordering, not bag-of-words.
    ordered = compute_rouge(["a b"], ["b a"])
    assert abs(ordered["rougeL"]) < 1e-6, ordered
    assert ordered["rouge1"] > 0.0, ordered

    # 7) Summary formatting matches the paper's R1/R2/RL ordering.
    summary = rouge_metric_summary({"rouge1": 42.1, "rouge2": 20.3, "rougeL": 39.4})
    assert summary == "ROUGE-1: 42.1  ROUGE-2: 20.3  ROUGE-L: 39.4", summary

    print("[cnndm] self-test passed")
    return True


if __name__ == "__main__":  # pragma: no cover
    _self_test()
