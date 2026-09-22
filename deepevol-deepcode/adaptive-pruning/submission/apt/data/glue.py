"""GLUE data pipeline for APT.

Paper references
----------------
* Sec. 5.1 (Tasks): "For BERT, RoBERTa, and T5 models, we train and evaluate on SST2
  and MNLI datasets from the GLUE benchmark (Wang et al., 2019) and report the dev
  set accuracy."  Table 7/8 additionally use the remaining GLUE tasks (QNLI, QQP,
  MRPC, CoLA, RTE, STSB) for the GLUE-average comparison.
* Appendix A (Table 6): GLUE hyper-parameters are split into *big* tasks
  (MNLI, SST2, QNLI, QQP) and *small* tasks (MRPC, CoLA, RTE, STSB), following CoFi
  (Xia et al., 2022)::

        Hyperparameter   GLUE-small   GLUE-big   SQuAD   CNN/DM   Alpaca
        Learning rate    2e-4         2e-4       2e-4    1e-4     1e-4
        Batch size       32           32         32      16       32
        Epochs           40           40         40      16       15
        Distill epochs   20           20         20      6        -

  Also: "To adaptively increase the tuning parameters in the LM, at the start of
  fine-tuning, we initialize adapter ranks to 8, with salient layers' ranks linearly
  increased.  The scaling factors are set as 2 statically."

This module provides everything the training scripts need for GLUE:

* task registry / metadata (num labels, sentence keys, metric, big vs small),
* the Table-6 hyper-parameters,
* tokenisation of the raw GLUE splits (single- and two-sentence tasks),
* ``torch.utils.data.Dataset`` wrappers plus a dynamic-padding collator
  (classification-style for BERT/RoBERTa, and a text-to-text target variant for T5),
* the paper's reported metrics (accuracy for MNLI/SST2/QNLI/RTE, F1 for QQP/MRPC,
  Matthews correlation for CoLA, Spearman correlation for STSB).

Heavy dependencies (``datasets``, ``transformers``, ``sklearn``) are imported lazily so
that the module can be imported and unit-tested without them installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GLUE_BIG_TASKS",
    "GLUE_SMALL_TASKS",
    "GLUE_TASKS",
    "MAX_SEQ_LENGTH",
    "TABLE6",
    "HParams",
    "GlueTaskSpec",
    "TASK_SPECS",
    "canonical_task_name",
    "is_big_task",
    "get_task_spec",
    "get_hparams",
    "num_labels",
    "collator_for_task",
    "T5_LABEL_TEXT",
    "GlueDataset",
    "T5GlueDataset",
    "GlueCollator",
    "T5GlueCollator",
    "build_glue_dataset",
    "build_glue_datasets",
    "make_glue_dataloaders",
    "compute_glue_metrics",
    "accuracy",
    "binary_f1",
    "matthews_corrcoef",
    "spearman_correlation",
    "pareto_big_small_order",
]


# --------------------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------------------

GLUE_BIG_TASKS: Tuple[str, ...] = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS: Tuple[str, ...] = ("mrpc", "cola", "rte", "stsb")
GLUE_TASKS: Tuple[str, ...] = GLUE_BIG_TASKS + GLUE_SMALL_TASKS

#: Maximum sequence length used for all GLUE fine-tuning runs.
MAX_SEQ_LENGTH = 128

_TASK_ALIASES: Dict[str, str] = {
    "mnli-m": "mnli",
    "mnli-mm": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatched": "mnli",
    "mnli": "mnli",
    "sst2": "sst2",
    "sst-2": "sst2",
    "qnli": "qnli",
    "qqp": "qqp",
    "mrpc": "mrpc",
    "cola": "cola",
    "rte": "rte",
    "stsb": "stsb",
    "sts-b": "stsb",
}


@dataclass
class GlueTaskSpec:
    """Static description of one GLUE task."""

    name: str
    hf_name: str
    num_labels: int
    metric: str
    sentence_keys: Tuple[str, ...]
    size: str
    label_names: Tuple[str, ...] = ()
    is_regression: bool = False
    max_seq_length: int = MAX_SEQ_LENGTH
    test_split: str = "validation"
    problem_type: str = "single_label_classification"

    @property
    def is_big(self) -> bool:
        return self.size == "big"

    @property
    def num_sentences(self) -> int:
        return len(self.sentence_keys)

    def label_text(self, label: int) -> str:
        """Target string used for text-to-text (T5) fine-tuning."""
        texts = T5_LABEL_TEXT.get(self.name)
        if texts is not None and 0 <= int(label) < len(texts):
            return texts[int(label)]
        return str(int(label))


TASK_SPECS: Dict[str, GlueTaskSpec] = {
    "mnli": GlueTaskSpec(
        name="mnli",
        hf_name="mnli",
        num_labels=3,
        metric="accuracy",
        sentence_keys=("premise", "hypothesis"),
        size="big",
        label_names=("entailment", "neutral", "contradiction"),
    ),
    "sst2": GlueTaskSpec(
        name="sst2",
        hf_name="sst2",
        num_labels=2,
        metric="accuracy",
        sentence_keys=("sentence",),
        size="big",
        label_names=("negative", "positive"),
    ),
    "qnli": GlueTaskSpec(
        name="qnli",
        hf_name="qnli",
        num_labels=2,
        metric="accuracy",
        sentence_keys=("question", "sentence"),
        size="big",
        label_names=("entailment", "not_entailment"),
    ),
    "qqp": GlueTaskSpec(
        name="qqp",
        hf_name="qqp",
        num_labels=2,
        metric="f1",
        sentence_keys=("question1", "question2"),
        size="big",
        label_names=("not_duplicate", "duplicate"),
    ),
    "mrpc": GlueTaskSpec(
        name="mrpc",
        hf_name="mrpc",
        num_labels=2,
        metric="f1",
        sentence_keys=("sentence1", "sentence2"),
        size="small",
        label_names=("not_equivalent", "equivalent"),
    ),
    "cola": GlueTaskSpec(
        name="cola",
        hf_name="cola",
        num_labels=2,
        metric="matthews",
        sentence_keys=("sentence",),
        size="small",
        label_names=("unacceptable", "acceptable"),
    ),
    "rte": GlueTaskSpec(
        name="rte",
        hf_name="rte",
        num_labels=2,
        metric="accuracy",
        sentence_keys=("sentence1", "sentence2"),
        size="small",
        label_names=("entailment", "not_entailment"),
    ),
    "stsb": GlueTaskSpec(
        name="stsb",
        hf_name="stsb",
        num_labels=1,
        metric="spearman",
        sentence_keys=("sentence1", "sentence2"),
        size="small",
        label_names=("label",),
        is_regression=True,
        problem_type="regression",
    ),
}


#: Target strings for T5 text-to-text GLUE fine-tuning (CoFi / T5 convention).
T5_LABEL_TEXT: Dict[str, Tuple[str, ...]] = {
    "sst2": ("negative", "positive"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    "rte": ("entailment", "not_entailment"),
    "mrpc": ("not_equivalent", "equivalent"),
    "cola": ("unacceptable", "acceptable"),
}

#: Prefixes prepended to T5 inputs (Mnli follows the CoFi/T5 convention of feeding the
#: two sentences; Sst2 uses the standard SST-2 template).
T5_INPUT_TEMPLATES: Dict[str, str] = {
    "sst2": "{sentence}",
    "mnli": "mnli premise: {premise} hypothesis: {hypothesis}",
    "qnli": "question: {question} sentence: {sentence}",
    "qqp": "question1: {question1} question2: {question2}",
    "rte": "sentence1: {sentence1} sentence2: {sentence2}",
    "mrpc": "sentence1: {sentence1} sentence2: {sentence2}",
    "cola": "{sentence}",
}


def canonical_task_name(task: str) -> str:
    """Map aliases (``SST-2``, ``mnli-mm``, ``STS-B`` ...) to the canonical key."""
    key = str(task).strip().lower().replace("_", "-")
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    key2 = key.replace("-", "")
    if key2 in TASK_SPECS:
        return key2
    raise KeyError(
        f"Unknown GLUE task '{task}'. Known tasks: {sorted(TASK_SPECS)}"
    )


def is_big_task(task: str) -> bool:
    """True for the GLUE-big split (MNLI, SST2, QNLI, QQP) of Appendix A."""
    return canonical_task_name(task) in GLUE_BIG_TASKS


def get_task_spec(task: str) -> GlueTaskSpec:
    return TASK_SPECS[canonical_task_name(task)]


def num_labels(task: str) -> int:
    return get_task_spec(task).num_labels


# --------------------------------------------------------------------------------------
# Appendix A, Table 6 hyper-parameters
# --------------------------------------------------------------------------------------


@dataclass
class HParams:
    """Training hyper-parameters from Appendix A, Table 6."""

    learning_rate: float
    batch_size: int
    epochs: int
    distill_epochs: int
    initial_rank: int = 8
    scaling: float = 2.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_seq_length: int = MAX_SEQ_LENGTH
    target_sparsity: float = 0.60
    name: str = ""

    @property
    def pruning_epochs(self) -> int:
        """Stage-1 (prune + self-distill) epochs."""
        return int(self.distill_epochs)

    @property
    def recovery_epochs(self) -> int:
        """Stage-2 (recover end task on the pruned LM) epochs."""
        return max(int(self.epochs) - int(self.distill_epochs), 0)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


TABLE6: Dict[str, HParams] = {
    "glue-small": HParams(
        learning_rate=2e-4,
        batch_size=32,
        epochs=40,
        distill_epochs=20,
        name="glue-small",
    ),
    "glue-big": HParams(
        learning_rate=2e-4,
        batch_size=32,
        epochs=40,
        distill_epochs=20,
        name="glue-big",
    ),
    "squad": HParams(
        learning_rate=2e-4,
        batch_size=32,
        epochs=40,
        distill_epochs=20,
        max_seq_length=384,
        name="squad",
    ),
    "cnndm": HParams(
        learning_rate=1e-4,
        batch_size=16,
        epochs=16,
        distill_epochs=6,
        max_seq_length=512,
        name="cnndm",
    ),
    "alpaca": HParams(
        learning_rate=1e-4,
        batch_size=32,
        epochs=15,
        distill_epochs=0,
        max_seq_length=512,
        name="alpaca",
    ),
}


def get_hparams(task: str, table: Optional[Dict[str, HParams]] = None) -> HParams:
    """Table-6 hyper-parameters for a GLUE task.

    ``task`` may be a GLUE task name (big/small split is resolved automatically) or a
    key of :data:`TABLE6` directly (``"glue-big"``, ``"squad"`` ...).
    """
    table = TABLE6 if table is None else table
    key = str(task).strip().lower().replace("_", "-")
    if key in table:
        return table[key]
    return table["glue-big" if is_big_task(task) else "glue-small"]


# --------------------------------------------------------------------------------------
# Raw data loading
# --------------------------------------------------------------------------------------


def _load_raw_dataset(
    task: str,
    split: str,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    use_datasets: bool = True,
) -> Any:
    """Load one GLUE split via the ``datasets`` library (or a local JSON fallback)."""
    spec = get_task_spec(task)
    if use_datasets:
        from datasets import load_dataset  # local import: optional dependency

        kwargs: Dict[str, Any] = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if data_dir:
            kwargs["data_dir"] = data_dir
        try:
            return load_dataset("glue", spec.hf_name, split=split, **kwargs)
        except Exception:  # pragma: no cover - network / revision dependent
            return load_dataset(
                "nyu-mll/glue", spec.hf_name, split=split, trust_remote_code=True,
                **kwargs,
            )
    # Offline fallback: a JSON-lines file with the raw GLUE columns.
    import json

    if not data_dir:
        raise ValueError("Offline mode requires `data_dir` pointing at the GLUE files.")
    path = os.path.join(data_dir, f"{spec.hf_name}.{split}.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _raw_columns(task: str, raw: Any) -> List[str]:
    """Column names of a HF dataset split (list-of-dicts friendly)."""
    if hasattr(raw, "column_names") and raw.column_names:
        return list(raw.column_names)
    if hasattr(raw, "features") and raw.features is not None:
        return list(raw.features.keys())
    if len(raw) > 0:
        return list(raw[0].keys())
    return []


def _raw_item(raw: Any, spec: GlueTaskSpec, index: int) -> Dict[str, Any]:
    """Fetch one example, preferring the task's sentence keys."""
    columns = _raw_columns(spec.name, raw)
    keys = [k for k in spec.sentence_keys if k in columns]
    if "idx" in columns:
        keys = keys + ["idx"]
    if "label" in columns:
        keys = keys + ["label"]
    return raw[index] if isinstance(raw, list) else {k: raw[k][index] for k in keys}


def _example_sentences(example: Dict[str, Any], spec: GlueTaskSpec) -> Tuple[str, ...]:
    """Extract the 1 or 2 input sentences from a raw example."""
    sentences: List[str] = []
    for key in spec.sentence_keys:
        value = example.get(key)
        if value is None:
            # Be forgiving about column naming differences (e.g. 'sent1'/'sent2').
            for alt in (key, key.replace("sentence", "sent"), key + "ence"):
                if alt in example:
                    value = example[alt]
                    break
        sentences.append("" if value is None else str(value))
    return tuple(sentences)


def _example_label(example: Dict[str, Any]) -> float:
    label = example.get("label")
    if label is None:
        label = example.get("labels", 0)
    return label if isinstance(label, (int, float)) else 0 if label == "" else label


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class GlueDataset:
    """Tokenised GLUE split for encoder models (BERT / RoBERTa).

    Returns dictionaries with ``input_ids``, ``attention_mask`` (and ``token_type_ids``
    when the tokenizer produces them) plus ``labels``.
    """

    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        task: str,
        tokenizer: Any,
        max_seq_length: int = MAX_SEQ_LENGTH,
        padding: str = "max_length",
        include_token_type_ids: bool = True,
        text_to_text: bool = False,
        max_target_length: int = 8,
    ) -> None:
        self.task = canonical_task_name(task)
        self.spec = get_task_spec(self.task)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.padding = padding
        self.include_token_type_ids = include_token_type_ids
        self.text_to_text = bool(text_to_text and self.task in T5_LABEL_TEXT)
        self.max_target_length = int(max_target_length)
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    # -- tokenisation -------------------------------------------------------------
    def _encode_inputs(self, sentences: Sequence[str]) -> Dict[str, Any]:
        spec = self.spec
        if len(sentences) == 2:
            encoded = self.tokenizer(
                sentences[0],
                sentences[1],
                max_length=self.max_seq_length,
                padding=self.padding,
                truncation=True,
                return_tensors=None,
            )
        else:
            if self.text_to_text and self.task in T5_INPUT_TEMPLATES:
                text = T5_INPUT_TEMPLATES[self.task].format(
                    sentence=sentences[0], sentence1=sentences[0], sentence2=""
                )
            else:
                text = sentences[0]
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding=self.padding,
                truncation=True,
                return_tensors=None,
            )
        out = {k: v for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        if self.include_token_type_ids and "token_type_ids" in encoded:
            out["token_type_ids"] = encoded["token_type_ids"]
        elif "token_type_ids" in encoded:
            # T5 has no segment embeddings; drop them.
            pass
        return out

    def _encode_target(self, label: float) -> Dict[str, Any]:
        text = self.spec.label_text(int(label))
        encoded = self.tokenizer(
            text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        ids = list(encoded["input_ids"])
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        labels = [tok if tok != pad_id else -100 for tok in ids]
        return {
            "labels": labels,
            "decoder_attention_mask": list(encoded.get("attention_mask", [1] * len(ids))),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        sentences = _example_sentences(example, self.spec)
        label = _example_label(example)
        item: Dict[str, Any] = self._encode_inputs(sentences)
        if self.text_to_text:
            item.update(self._encode_target(label))
            item["label_ids"] = int(label)
        else:
            if self.spec.is_regression:
                item["labels"] = float(label)
            else:
                item["labels"] = int(label)
        return item


class T5GlueDataset(GlueDataset):
    """GLUE split prepared for T5 text-to-text fine-tuning."""

    def __init__(self, examples, task, tokenizer, max_seq_length=MAX_SEQ_LENGTH,
                 max_target_length: int = 8, padding: str = "max_length") -> None:
        super().__init__(
            examples,
            task,
            tokenizer,
            max_seq_length=max_seq_length,
            padding=padding,
            include_token_type_ids=False,
            text_to_text=True,
            max_target_length=max_target_length,
        )


# --------------------------------------------------------------------------------------
# Collators
# --------------------------------------------------------------------------------------


class GlueCollator:
    """Dynamic (longest-in-batch) padding collator for encoder classification."""

    def __init__(
        self,
        tokenizer: Any = None,
        padding: str = "longest",
        label_pad_token_id: float = -100.0,
        keep_columns: Sequence[str] = ("input_ids", "attention_mask", "token_type_ids"),
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id
        self.keep_columns = tuple(keep_columns)

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            return int(self.tokenizer.pad_token_id)
        return 0

    def _pad(self, sequences: List[List[int]], value: int) -> Any:
        import torch

        if self.padding == "max_length":
            width = max(len(s) for s in sequences)
        else:  # 'longest' / 'do_not_pad'
            width = max(len(s) for s in sequences)
        rows = [list(s) + [value] * (width - len(s)) for s in sequences]
        return torch.tensor(rows, dtype=torch.long)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        batch: Dict[str, Any] = {}
        for column in self.keep_columns:
            if column in features[0]:
                batch[column] = self._pad(
                    [list(f[column]) for f in features],
                    self.pad_token_id if column != "token_type_ids" else 0,
                )
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            if isinstance(labels[0], float):
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
        for extra in ("idx", "label_ids"):
            if extra in features[0]:
                batch[extra] = torch.tensor([f[extra] for f in features], dtype=torch.long)
        return batch


class T5GlueCollator:
    """Dynamic padding collator for T5 GLUE text-to-text fine-tuning."""

    def __init__(
        self,
        tokenizer: Any = None,
        padding: str = "longest",
        label_pad_token_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            return int(self.tokenizer.pad_token_id)
        return 0

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        enc = GlueCollator(self.tokenizer, padding=self.padding,
                           keep_columns=("input_ids", "attention_mask"))(features)
        max_len = max(len(f["labels"]) for f in features)
        labels = [
            list(f["labels"]) + [self.label_pad_token_id] * (max_len - len(f["labels"]))
            for f in features
        ]
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        if "decoder_attention_mask" in features[0]:
            widths = max(len(f["decoder_attention_mask"]) for f in features)
            masks = [
                list(f["decoder_attention_mask"]) + [0] * (widths - len(f["decoder_attention_mask"]))
                for f in features
            ]
            enc["decoder_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        if "label_ids" in features[0]:
            enc["label_ids"] = torch.tensor([f["label_ids"] for f in features], dtype=torch.long)
        return enc


def collator_for_task(task: str, tokenizer: Any = None, model_type: str = "encoder",
                      padding: str = "longest") -> Any:
    """Return the appropriate collator for a task / model family."""
    name = str(model_type).lower()
    t5_like = name in ("t5", "seq2seq", "text2text") or "t5" in name
    if t5_like and canonical_task_name(task) in T5_LABEL_TEXT:
        return T5GlueCollator(tokenizer, padding=padding)
    return GlueCollator(tokenizer, padding=padding)


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def build_glue_dataset(
    task: str,
    tokenizer: Any,
    split: str = "validation",
    max_seq_length: int = MAX_SEQ_LENGTH,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    model_type: str = "encoder",
    text_to_text: Optional[bool] = None,
    use_datasets: bool = True,
    padding: str = "max_length",
) -> GlueDataset:
    """Load and tokenise a single GLUE split."""
    spec = get_task_spec(task)
    raw = _load_raw_dataset(spec.name, split, cache_dir=cache_dir, data_dir=data_dir,
                            use_datasets=use_datasets)
    examples: List[Dict[str, Any]] = []
    for index in range(len(raw)):
        examples.append(_raw_item(raw, spec, index))
    if text_to_text is None:
        text_to_text = "t5" in str(model_type).lower()
    if text_to_text and spec.name in T5_LABEL_TEXT:
        return T5GlueDataset(examples, spec.name, tokenizer,
                             max_seq_length=max_seq_length, padding=padding)
    return GlueDataset(examples, spec.name, tokenizer,
                       max_seq_length=max_seq_length, padding=padding)


def build_glue_datasets(
    task: str,
    tokenizer: Any,
    splits: Sequence[str] = ("train", "validation"),
    max_seq_length: int = MAX_SEQ_LENGTH,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    model_type: str = "encoder",
    use_datasets: bool = True,
    padding: str = "max_length",
) -> Dict[str, GlueDataset]:
    """Load several splits of a GLUE task (``{"train": ..., "validation": ...}``)."""
    spec = get_task_spec(task)
    out: Dict[str, GlueDataset] = {}
    for split in splits:
        name = spec.test_split if split in ("dev", "test") else split
        out[split] = build_glue_dataset(
            spec.name,
            tokenizer,
            split=name,
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            data_dir=data_dir,
            model_type=model_type,
            use_datasets=use_datasets,
            padding=padding,
        )
    return out


def make_glue_dataloaders(
    task: str,
    tokenizer: Any,
    batch_size: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    model_type: str = "encoder",
    splits: Sequence[str] = ("train", "validation"),
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    use_datasets: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience helper: build tokenised splits and wrap them in DataLoaders."""
    import torch
    from torch.utils.data import DataLoader

    hparams = get_hparams(task)
    batch_size = hparams.batch_size if batch_size is None else int(batch_size)
    max_seq_length = hparams.max_seq_length if max_seq_length is None else int(max_seq_length)
    datasets = build_glue_datasets(
        task,
        tokenizer,
        splits=splits,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir,
        data_dir=data_dir,
        model_type=model_type,
        use_datasets=use_datasets,
        padding="longest" if kwargs.pop("dynamic_padding", False) else "max_length",
    )
    collator = collator_for_task(task, tokenizer, model_type=model_type)
    loaders: Dict[str, Any] = {}
    for split, dataset in datasets.items():
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train and split == "train",
            collate_fn=collator,
            num_workers=num_workers,
            generator=generator,
            **kwargs,
        )
    return loaders


# --------------------------------------------------------------------------------------
# Metrics (Sec. 5.1 / Table 2 & 8)
# --------------------------------------------------------------------------------------


def accuracy(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    import numpy as np

    preds = np.asarray(predictions).reshape(-1)
    refs = np.asarray(references).reshape(-1)
    if preds.shape != refs.shape:
        raise ValueError(f"shape mismatch: {preds.shape} vs {refs.shape}")
    return float((preds == refs).mean()) if refs.size else 0.0


def binary_f1(predictions: Sequence[Any], references: Sequence[Any],
              positive_label: int = 1) -> float:
    """Binary F1 used for MRPC (and QQP in the HuggingFace GLUE convention)."""
    import numpy as np

    preds = np.asarray(predictions).reshape(-1)
    refs = np.asarray(references).reshape(-1)
    tp = float(((preds == positive_label) & (refs == positive_label)).sum())
    fp = float(((preds == positive_label) & (refs != positive_label)).sum())
    fn = float(((preds != positive_label) & (refs == positive_label)).sum())
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def matthews_corrcoef(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    """Matthews correlation for CoLA (implemented locally, sklearn optional)."""
    try:
        from sklearn.metrics import matthews_corrcoef as _mcc

        return float(_mcc(references, predictions))
    except Exception:  # pragma: no cover - sklearn missing
        import numpy as np

        preds = np.asarray(predictions).reshape(-1)
        refs = np.asarray(references).reshape(-1)
        tp = float(((preds == 1) & (refs == 1)).sum())
        tn = float(((preds == 0) & (refs == 0)).sum())
        fp = float(((preds == 1) & (refs == 0)).sum())
        fn = float(((preds == 0) & (refs == 1)).sum())
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def spearman_correlation(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    """Spearman correlation for STSB."""
    try:
        from scipy.stats import spearmanr

        value = spearmanr(references, predictions).correlation
        return float(value)
    except Exception:  # pragma: no cover
        import numpy as np

        preds = np.asarray(predictions, dtype=float).reshape(-1)
        refs = np.asarray(references, dtype=float).reshape(-1)
        if preds.size < 2:
            return 0.0

        def _rank(values):
            order = values.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(values), dtype=float)
            return ranks

        rp, rr = _rank(preds), _rank(refs)
        rp = rp - rp.mean()
        rr = rr - rr.mean()
        denom = float((rp**2).sum() ** 0.5 * (rr**2).sum() ** 0.5)
        return float((rp * rr).sum() / denom) if denom > 0 else 0.0


_METRIC_FUNCTIONS = {
    "accuracy": accuracy,
    "f1": binary_f1,
    "matthews": matthews_corrcoef,
    "spearman": spearman_correlation,
}


def compute_glue_metrics(task: str, predictions: Sequence[Any],
                         references: Sequence[Any]) -> Dict[str, float]:
    """Metric(s) reported by the paper for a GLUE task.

    The primary metric follows the HuggingFace GLUE convention used by the paper:
    accuracy for MNLI/SST2/QNLI/RTE, F1 for QQP/MRPC, Matthews for CoLA, Spearman for
    STSB.
    """
    spec = get_task_spec(task)
    fn = _METRIC_FUNCTIONS[spec.metric]
    metrics = {spec.metric: float(fn(predictions, references))}
    if spec.metric != "accuracy" and not spec.is_regression:
        metrics["accuracy"] = accuracy(predictions, references)
    return metrics


def pareto_big_small_order(tasks: Iterable[str]) -> List[str]:
    """Order tasks as in Appendix A: GLUE-big first, then GLUE-small."""
    names = [canonical_task_name(t) for t in tasks]
    big = [t for t in GLUE_BIG_TASKS if t in names]
    small = [t for t in GLUE_SMALL_TASKS if t in names]
    other = [t for t in names if t not in big and t not in small]
    return big + small + other


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover - manual check
    assert canonical_task_name("SST-2") == "sst2"
    assert canonical_task_name("mnli-mm") == "mnli"
    assert canonical_task_name("STS-B") == "stsb"
    assert is_big_task("MNLI") and not is_big_task("MRPC")

    cos = get_hparams("mnli")
    assert (cos.learning_rate, cos.batch_size, cos.epochs, cos.distill_epochs) == (
        2e-4, 32, 40, 20)
    assert cos.initial_rank == 8 and cos.scaling == 2.0
    small = get_hparams("rte")
    assert small.batch_size == 32 and small.epochs == 40 and small.distill_epochs == 20
    assert TABLE6["cnndm"].learning_rate == 1e-4 and TABLE6["cnndm"].distill_epochs == 6
    assert TABLE6["squad"].epochs == 40

    assert num_labels("mnli") == 3 and num_labels("stsb") == 1
    assert get_task_spec("qqp").sentence_keys == ("question1", "question2")

    preds = [1, 1, 0, 0, 1]
    refs = [1, 0, 0, 1, 1]
    assert abs(accuracy(preds, refs) - 0.6) < 1e-9
    assert abs(binary_f1(preds, refs) - (2 * (2 / 3) * (2 / 3) / (4 / 3))) < 1e-9
    assert abs(compute_glue_metrics("mnli", preds, refs)["accuracy"] - 0.6) < 1e-9
    assert "f1" in compute_glue_metrics("mrpc", preds, refs)
    assert "spearman" in compute_glue_metrics("stsb", [0.1, 0.2, 0.3], [0.2, 0.1, 0.4])
    assert -1.0 <= matthews_corrcoef(preds, refs) <= 1.0

    # Dataset / collator round-trip with a fake word-level tokenizer (no transformers).
    class _FakeTokenizer:
        pad_token_id = 0

        def _ids(self, text: str):
            return [1 + (abs(hash(w)) % 999) for w in str(text).split()] or [1]

        def __call__(self, a, b=None, max_length=None, padding="max_length",
                     truncation=True, return_tensors=None):
            ids = self._ids(a) if b is None else self._ids(a) + [2] + self._ids(b)
            ids = ids[:max_length]
            if padding == "max_length":
                ids = ids + [0] * (max_length - len(ids))
            return {"input_ids": ids, "attention_mask": [1 if i != 0 else 0 for i in ids],
                    "token_type_ids": [0] * len(ids)}

    raw = [{"premise": "a b c", "hypothesis": "d e", "label": 1},
           {"premise": "f", "hypothesis": "g h i j", "label": 0}]
    tok = _FakeTokenizer()
    ds = GlueDataset(raw, "mnli", tok, max_seq_length=16)
    assert len(ds) == 2 and len(ds[0]["input_ids"]) == 16
    collator = GlueCollator(tok)
    batch = collator([ds[0], ds[1]])
    assert tuple(batch["input_ids"].shape) == (2, 16)
    assert tuple(batch["labels"].shape) == (2,)

    t5ds = T5GlueDataset(raw, "mnli", tok, max_seq_length=8, max_target_length=4)
    assert t5ds[0]["labels"][0] == 1  # 'entailment' -> fake ids, no padding removed
    t5batch = T5GlueCollator(tok)([t5ds[0], t5ds[1]])
    assert t5batch["labels"].shape[0] == 2

    assert pareto_big_small_order(["mrpc", "mnli", "sst2"]) == ["mnli", "sst2", "mrpc"]
    print("apt.data.glue self-test passed.")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
