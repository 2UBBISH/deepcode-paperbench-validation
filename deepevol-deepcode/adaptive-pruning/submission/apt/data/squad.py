"""SQuAD v2.0 data pipeline for APT (paper Sec. 5.1, Appendix A Table 6).

The paper trains and evaluates ``RoBERTa_base`` on SQuAD v2.0 (Rajpurkar et al.,
2018) and reports the dev-set **F1** score (Table 2).  Table 6 gives the
hyper-parameters used for the SQuAD runs::

    Learning rate   2e-4
    Batch size      32
    Epochs          40
    Distill epochs  20

This module implements everything needed to run those experiments:

* loading SQuAD v2.0 (HuggingFace ``datasets`` with an offline JSONL fallback),
* feature extraction with a sliding window (``doc_stride``) over the context,
* answer-span labelling (``start_positions`` / ``end_positions``) with the
  unanswerable-question convention ``start = end = cls_index`` (SQuAD v2.0),
* dynamic-padding collators and ``DataLoader`` construction,
* span post-processing (``get_best_indexes`` / ``write_predictions``) plus the
  official SQuAD v2.0 ``exact match`` / ``F1`` metrics (n-best with no-answer
  handling), implemented in pure Python with an optional ``evaluate`` backend.

All heavy dependencies (``datasets``, ``transformers``, ``torch``, ``numpy``)
are imported lazily so the module can be imported — and ``_self_test`` run — in
a bare environment.
"""

from __future__ import annotations

import collections
import json
import os
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Defaults (HF run_qa + Table 6 from the paper)
# ---------------------------------------------------------------------------

SQUAD_MAX_SEQ_LENGTH = 384
SQUAD_DOC_STRIDE = 128
SQUAD_MAX_QUERY_LENGTH = 64
SQUAD_N_BEST_SIZE = 20
SQUAD_MAX_ANSWER_LENGTH = 30
SQUAD_NULL_SCORE_DIFF_THRESHOLD = 0.0
NO_ANSWER = "no answer"

# Table 6 rows used by the SQuAD experiments (mirrors apt/data/glue.py TABLE6).
SQUAD_TABLE6 = {
    "learning_rate": 2e-4,
    "batch_size": 32,
    "epochs": 40,
    "distill_epochs": 20,
    "target_sparsity": 0.60,
}


# ---------------------------------------------------------------------------
# Examples / features
# ---------------------------------------------------------------------------


@dataclass
class SquadExample:
    """A single (question, context, answer) triple."""

    qas_id: str
    question_text: str
    context_text: str
    answer_text: str = ""
    start_position_character: int = 0
    is_impossible: bool = False
    answers: List[str] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return bool(self.answers) and not self.is_impossible


@dataclass
class SquadFeatures:
    """Tokenised feature (one sliding window of one example)."""

    unique_id: int
    example_index: int
    qas_id: str
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    cls_index: int
    p_mask: List[float]
    offset_mapping: List[Tuple[int, int]]
    start_position: int
    end_position: int
    span_index: int = 0
    token_is_max_context: Dict[int, bool] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "example_index": self.example_index,
            "qas_id": self.qas_id,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "token_type_ids": list(self.token_type_ids),
            "cls_index": self.cls_index,
            "p_mask": list(self.p_mask),
            "offset_mapping": list(self.offset_mapping),
            "start_position": self.start_position,
            "end_position": self.end_position,
            "span_index": self.span_index,
        }


@dataclass
class SquadPrediction:
    """Post-processed prediction for one feature (n-best spans)."""

    unique_id: int
    example_index: int
    span_index: int
    start_index: int
    end_index: int
    start_logit: float
    end_logit: float
    score: float
    text: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def canonical_split(split: str) -> str:
    """Map common split aliases onto SQuAD v2.0 split names."""

    key = str(split).lower()
    if key in {"train", "training"}:
        return "train"
    if key in {"dev", "valid", "validation", "eval"}:
        return "validation"
    if key == "test":
        return "test"
    raise KeyError(f"unknown SQuAD split: {split!r}")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_squad_split(
    split: str = "validation",
    *,
    data_dir: Optional[str] = None,
    hf_name: str = "squad_v2",
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load one SQuAD v2.0 split as a list of raw records.

    Tries HuggingFace ``datasets`` first (with a ``rajpurkar/squad_v2``
    fallback), then an offline JSONL/JSON file ``<data_dir>/<split>.jsonl`` or
    the raw SQuAD ``.json`` structure.
    """

    split = canonical_split(split)

    if use_datasets:
        try:  # pragma: no cover - depends on environment
            from datasets import load_dataset  # type: ignore

            last_error: Optional[BaseException] = None
            for candidate in (hf_name, "rajpurkar/squad_v2"):
                try:
                    dataset = load_dataset(candidate, split=split, cache_dir=cache_dir)
                    return [dict(row) for row in dataset]
                except BaseException as exc:  # noqa: BLE001 - fall through
                    last_error = exc
            if last_error is not None and data_dir is None:
                raise last_error
        except BaseException:  # noqa: BLE001
            if data_dir is None:
                raise

    if data_dir is None:
        raise ValueError(
            "cannot load SQuAD: `data_dir` is required when the HuggingFace "
            "`datasets` backend is unavailable"
        )

    jsonl_path = os.path.join(data_dir, f"{split}.jsonl")
    if os.path.exists(jsonl_path):
        return _load_jsonl(jsonl_path)

    json_path = os.path.join(data_dir, f"{split}-v2.0.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(data_dir, f"squad_v2_{split}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return squad_records_from_raw(raw)

    raise FileNotFoundError(
        f"no SQuAD file found for split {split!r} under {data_dir!r}"
    )


def squad_records_from_raw(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the raw SQuAD JSON structure into one record per question."""

    records: List[Dict[str, Any]] = []
    for article in raw.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", []) or []
                records.append(
                    {
                        "id": qa.get("id", ""),
                        "title": article.get("title", ""),
                        "context": context,
                        "question": qa.get("question", ""),
                        "answers": {
                            "text": [a.get("text", "") for a in answers],
                            "answer_start": [a.get("answer_start", -1) for a in answers],
                        },
                        "is_impossible": bool(qa.get("is_impossible", len(answers) == 0)),
                    }
                )
    return records


def _answers_to_lists(answers: Any) -> Tuple[List[str], List[int]]:
    if answers is None:
        return [], []
    if isinstance(answers, dict):
        return list(answers.get("text", []) or []), list(
            answers.get("answer_start", []) or []
        )
    texts, starts = [], []
    if isinstance(answers, (list, tuple)):
        for answer in answers:
            if isinstance(answer, dict):
                texts.append(answer.get("text", ""))
                starts.append(int(answer.get("answer_start", -1)))
            else:  # pragma: no cover - unusual schema
                texts.append(str(answer))
                starts.append(-1)
    return texts, starts


def squad_examples_from_records(
    records: Iterable[Dict[str, Any]], *, is_training: bool = True
) -> List[SquadExample]:
    """Convert raw records into :class:`SquadExample` objects."""

    examples: List[SquadExample] = []
    for index, row in enumerate(records):
        qas_id = str(row.get("id", index))
        answers = row.get("answers")
        texts, starts = _answers_to_lists(answers)
        if not texts and row.get("answer", None) is not None:  # single-answer schema
            texts, starts = [str(row["answer"])], [int(row.get("answer_start", -1))]

        is_impossible = bool(row.get("is_impossible", False))
        context = row.get("context", row.get("context_text", ""))

        # Skip answers that cannot be located in the context (standard run_qa).
        answer_text, start_char = "", 0
        if is_training and not is_impossible:
            if not texts:
                is_impossible = True
            else:
                found = False
                for text, start in zip(texts, starts):
                    if text and start >= 0 and start < len(context) and context[
                        start : start + len(text)
                    ] == text:
                        answer_text, start_char, found = text, int(start), True
                        break
                if not found:
                    continue

        examples.append(
            SquadExample(
                qas_id=qas_id,
                question_text=row.get("question", row.get("question_text", "")),
                context_text=context,
                answer_text=answer_text,
                start_position_character=start_char,
                is_impossible=is_impossible,
                answers=list(texts),
            )
        )
    return examples


def load_squad_examples(
    split: str = "validation",
    *,
    is_training: Optional[bool] = None,
    data_dir: Optional[str] = None,
    hf_name: str = "squad_v2",
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[List[SquadExample], List[Dict[str, Any]]]:
    """Load a split and return ``(examples, raw_records)``."""

    split = canonical_split(split)
    records = load_squad_split(
        split,
        data_dir=data_dir,
        hf_name=hf_name,
        use_datasets=use_datasets,
        cache_dir=cache_dir,
    )
    training = split == "train" if is_training is None else bool(is_training)
    return squad_examples_from_records(records, is_training=training), records


# ---------------------------------------------------------------------------
# Tokenisation / feature extraction
# ---------------------------------------------------------------------------


def _is_whitespace(char: str) -> bool:
    return char in " \t\n\r" or char == "\u202f" or char == "\xa0"


def get_doc_spans(
    doc_tokens: Sequence[Any],
    max_tokens_for_doc: int,
    doc_stride: int,
) -> Iterator[Tuple[int, int, Dict[int, bool]]]:
    """Yield ``(start, end, token_is_max_context)`` sliding windows.

    ``doc_tokens`` may be strings or ``(token, offset)`` pairs; only the length
    of the sequence matters here.
    """

    total = len(doc_tokens)
    if max_tokens_for_doc <= 0:
        return
    if total <= max_tokens_for_doc:
        yield 0, total, {i: True for i in range(total)}
        return

    max_context: Dict[int, bool] = {}
    for start in range(0, total, doc_stride):
        end = min(start + max_tokens_for_doc, total)
        if start >= total:
            break
        for index in range(start, end):
            score = min(end - index, index - start + 1)
            previous = max_context.get(index)
            if previous is None or score > previous:
                max_context[index] = score  # type: ignore[assignment]
        yield start, end, {i: True for i in range(total)}
        if end == total:
            break

    # mark non-best offsets as False (max-context criterion)
    for start in range(0, total, doc_stride):
        end = min(start + max_tokens_for_doc, total)
        for index in range(start, end):
            best = max_context.get(index, 0)
            score = min(end - index, index - start + 1)
            if score < best:  # type: ignore[operator]
                max_context[index] = False
    yield -1, -1, dict(max_context)  # sentinel handled by caller (ignored)


def _doc_spans(
    n_tokens: int, max_tokens_for_doc: int, doc_stride: int
) -> List[Tuple[int, int, Dict[int, bool]]]:
    """Clean sliding-window generation with ``max_context`` bookkeeping."""

    spans: List[Tuple[int, int, Dict[int, bool]]] = []
    if max_tokens_for_doc <= 0:
        return spans
    if n_tokens <= max_tokens_for_doc:
        return [(0, n_tokens, {i: True for i in range(n_tokens)})]

    best_scores: Dict[int, int] = {}
    for start in range(0, n_tokens, doc_stride):
        end = min(start + max_tokens_for_doc, n_tokens)
        if start >= end:
            break
        for index in range(start, end):
            score = min(end - index, index - start + 1)
            if score > best_scores.get(index, -1):
                best_scores[index] = score
        if end == n_tokens:
            break

    for start in range(0, n_tokens, doc_stride):
        end = min(start + max_tokens_for_doc, n_tokens)
        if start >= end:
            break
        flags = {}
        for index in range(start, end):
            score = min(end - index, index - start + 1)
            flags[index] = score >= best_scores.get(index, 0)
        spans.append((start, end, flags))
        if end == n_tokens:
            break
    return spans


def _find_answer_offsets(
    context: str, answer_text: str, start_char: int
) -> Tuple[int, int]:
    """Character (start, end) of the answer inside the context."""

    if not answer_text:
        return start_char, start_char
    start_char = max(0, int(start_char))
    end_char = start_char + len(answer_text)
    if context[start_char : start_char + len(answer_text)] != answer_text:
        # RoBERTa-style tokenisers may lose the leading space.
        stripped = answer_text.lstrip()
        if stripped and context[start_char : start_char + len(stripped)] == stripped:
            answer_text = stripped
            end_char = start_char + len(answer_text)
    return start_char, end_char


def convert_examples_to_features(
    examples: Sequence[SquadExample],
    tokenizer: Any,
    max_seq_length: int = SQUAD_MAX_SEQ_LENGTH,
    doc_stride: int = SQUAD_DOC_STRIDE,
    max_query_length: int = SQUAD_MAX_QUERY_LENGTH,
    is_training: bool = True,
    return_token_type_ids: bool = True,
    add_special_tokens: bool = True,
    pad_to_max_length: bool = False,
    pad_token_id: int = 0,
    unique_id_start: int = 1000000000,
    verbose: bool = True,
) -> List[SquadFeatures]:
    """Tokenise examples into sliding-window :class:`SquadFeatures`."""

    if max_query_length > max_seq_length:
        max_query_length = max_seq_length

    special_tokens = 3  # [CLS] question [SEP] context [SEP] (or <s> ... </s>)
    max_tokens_for_doc = max_seq_length - max_query_length - special_tokens
    if max_tokens_for_doc <= 0:
        max_tokens_for_doc = max(1, max_seq_length // 2)

    pad_id = int(
        getattr(tokenizer, "pad_token_id", None)
        if getattr(tokenizer, "pad_token_id", None) is not None
        else pad_token_id
    )
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", pad_id)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    if cls_id is None:
        cls_id = getattr(tokenizer, "bos_token_id", sep_id)

    def _encode(text: str, offsets: bool) -> Tuple[List[int], List[Tuple[int, int]]]:
        kwargs: Dict[str, Any] = {
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        }
        if offsets:
            kwargs["return_offsets_mapping"] = True
        try:
            encoded = tokenizer(text, **kwargs)
        except TypeError:  # slow tokenizer without offset mapping
            kwargs.pop("return_offsets_mapping", None)
            encoded = tokenizer(text, **kwargs)
            tokens = tokenizer.tokenize(text)
            # approximate offsets by cumulative character positions
            approx, cursor = [], 0
            for token in tokens:
                clean = token.replace("\u0120", " ").replace("\u2581", " ")
                found = text.find(clean.strip(), cursor) if clean.strip() else cursor
                if found < 0:
                    found = cursor
                start = found
                end = start + max(len(clean.strip()), 1)
                approx.append((start, end))
                cursor = end
            return list(encoded["input_ids"]), approx

        ids = list(encoded["input_ids"])
        if offsets:
            mapping = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
        else:
            mapping = [(0, 0)] * len(ids)
        return ids, mapping

    features: List[SquadFeatures] = []
    unique_id = int(unique_id_start)

    for example_index, example in enumerate(examples):
        query_ids, _ = _encode(example.question_text, offsets=False)
        if len(query_ids) > max_query_length:
            if max_query_length > 1:
                query_ids = query_ids[-max_query_length + 1 :]
            else:
                query_ids = query_ids[:max_query_length]

        context = example.context_text
        context_ids, context_offsets = _encode(context, offsets=True)
        n_context = len(context_ids)

        answer_start_char, answer_end_char = _find_answer_offsets(
            context, example.answer_text, example.start_position_character
        )

        spans = _doc_spans(n_context, max_tokens_for_doc, doc_stride)
        if not spans:
            spans = [(0, min(n_context, max_tokens_for_doc) or 1,
                      {i: True for i in range(min(n_context, max_tokens_for_doc) or 1)})]

        for span_index, (span_start, span_end, max_context) in enumerate(spans):
            span_ids = list(context_ids[span_start:span_end])
            span_offsets = [
                (min(len(context), max(0, s)), min(len(context), max(0, e)))
                for s, e in context_offsets[span_start:span_end]
            ]

            token_is_max_context = {
                span_start + i: bool(flag) for i, flag in max_context.items()
            }

            # ---- answer span labelling -------------------------------------
            cls_index = 0
            start_position = cls_index
            end_position = cls_index
            if is_training:
                if example.is_impossible or not example.answer_text:
                    start_position = cls_index
                    end_position = cls_index
                else:
                    start_in_span = None
                    end_in_span = None
                    for i, (char_start, char_end) in enumerate(span_offsets):
                        if char_start == char_end:
                            continue
                        if char_start <= answer_start_char < char_end:
                            start_in_span = i
                        if char_start < answer_end_char <= char_end:
                            end_in_span = i
                            break
                    if start_in_span is None:
                        for i, (char_start, char_end) in enumerate(span_offsets):
                            if char_start >= answer_start_char:
                                start_in_span = i
                                break
                    if end_in_span is None:
                        end_in_span = start_in_span
                    if start_in_span is None or end_in_span is None:
                        # answer not contained in this window
                        continue
                    if end_in_span < start_in_span:
                        start_in_span, end_in_span = end_in_span, start_in_span
                    start_position = start_in_span + 1 + len(query_ids)
                    end_position = end_in_span + 1 + len(query_ids)
                    if start_position >= max_seq_length or end_position >= max_seq_length:
                        continue

            # ---- assemble the sequence -------------------------------------
            input_ids = [int(cls_id)] + [int(i) for i in query_ids] + [int(sep_id)]
            offset_mapping: List[Tuple[int, int]] = [(0, 0)] * len(input_ids)
            p_mask = [1.0] * len(input_ids)  # 1 == masked out from span scoring
            offset_mapping += span_offsets
            p_mask += [0.0] * len(span_offsets)
            input_ids += [int(i) for i in span_ids]
            input_ids.append(int(sep_id))
            offset_mapping.append((0, 0))
            p_mask.append(1.0)

            token_type_ids = [0] * (len(query_ids) + 2) + [1] * (len(span_ids) + 1)
            attention_mask = [1] * len(input_ids)

            if pad_to_max_length and len(input_ids) < max_seq_length:
                pad_length = max_seq_length - len(input_ids)
                input_ids += [pad_id] * pad_length
                attention_mask += [0] * pad_length
                token_type_ids += [0] * pad_length
                offset_mapping += [(0, 0)] * pad_length
                p_mask += [1.0] * pad_length

            features.append(
                SquadFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    qas_id=example.qas_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    offset_mapping=offset_mapping,
                    start_position=start_position,
                    end_position=end_position,
                    span_index=span_index,
                    token_is_max_context=token_is_max_context,
                )
            )
            unique_id += 1

    if verbose:
        print(
            f"[squad] tokenised {len(examples)} examples -> {len(features)} features"
        )
    return features


# ---------------------------------------------------------------------------
# Datasets / collators / dataloaders
# ---------------------------------------------------------------------------


class SquadDataset:
    """Torch-style dataset over :class:`SquadFeatures`."""

    def __init__(self, features: Sequence[SquadFeatures], training: bool = True):
        self.features = list(features)
        self.training = training

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        feature = self.features[index]
        item: Dict[str, Any] = {
            "input_ids": feature.input_ids,
            "attention_mask": feature.attention_mask,
            "token_type_ids": feature.token_type_ids,
            "cls_index": feature.cls_index,
            "p_mask": feature.p_mask,
            "offset_mapping": feature.offset_mapping,
            "unique_id": feature.unique_id,
            "example_index": feature.example_index,
            "span_index": feature.span_index,
        }
        if self.training:
            item["start_positions"] = feature.start_position
            item["end_positions"] = feature.end_position
        return item


class SquadCollator:
    """Pad a batch of variable-length span features (labels stay scalars)."""

    def __init__(
        self,
        tokenizer: Any = None,
        pad_token_id: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        include_token_type_ids: bool = True,
    ):
        self.tokenizer = tokenizer
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            pad_token_id = int(tokenizer.pad_token_id)
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.include_token_type_ids = include_token_type_ids

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch  # noqa: WPS433 (lazy)

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        p_mask = torch.ones((batch_size, max_len), dtype=torch.float)

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(
                feature["attention_mask"], dtype=torch.long
            )
            token_type_ids[row, :length] = torch.tensor(
                feature["token_type_ids"], dtype=torch.long
            )
            p_mask[row, :length] = torch.tensor(feature["p_mask"], dtype=torch.float)
            p_mask[row, length:] = 1.0

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "p_mask": p_mask,
        }
        if self.include_token_type_ids:
            batch["token_type_ids"] = token_type_ids

        if "start_positions" in features[0]:
            batch["start_positions"] = torch.tensor(
                [int(f["start_positions"]) for f in features], dtype=torch.long
            )
            batch["end_positions"] = torch.tensor(
                [int(f["end_positions"]) for f in features], dtype=torch.long
            )
        for key in ("cls_index", "unique_id", "example_index", "span_index"):
            if key in features[0]:
                batch[key] = torch.tensor(
                    [int(f[key]) for f in features], dtype=torch.long
                )
        return batch


def build_squad_features(
    split: str = "validation",
    tokenizer: Any = None,
    *,
    max_seq_length: int = SQUAD_MAX_SEQ_LENGTH,
    doc_stride: int = SQUAD_DOC_STRIDE,
    max_query_length: int = SQUAD_MAX_QUERY_LENGTH,
    is_training: Optional[bool] = None,
    data_dir: Optional[str] = None,
    hf_name: str = "squad_v2",
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
    pad_to_max_length: bool = False,
    unique_id_start: int = 1000000000,
    verbose: bool = False,
) -> Tuple[List[SquadFeatures], List[SquadExample], List[Dict[str, Any]]]:
    """Load, tokenise and return ``(features, examples, raw_records)``."""

    if tokenizer is None:
        raise ValueError("`tokenizer` is required to build SQuAD features")

    examples, records = load_squad_examples(
        split,
        is_training=is_training,
        data_dir=data_dir,
        hf_name=hf_name,
        use_datasets=use_datasets,
        cache_dir=cache_dir,
    )
    training = canonical_split(split) == "train" if is_training is None else bool(is_training)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=training,
        pad_to_max_length=pad_to_max_length,
        unique_id_start=unique_id_start,
        verbose=verbose,
    )
    return features, examples, records


def make_squad_dataloaders(
    tokenizer: Any,
    splits: Sequence[str] = ("train", "validation"),
    *,
    batch_size: Optional[int] = None,
    max_seq_length: int = SQUAD_MAX_SEQ_LENGTH,
    doc_stride: int = SQUAD_DOC_STRIDE,
    max_query_length: int = SQUAD_MAX_QUERY_LENGTH,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    data_dir: Optional[str] = None,
    hf_name: str = "squad_v2",
    use_datasets: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build ``{split: DataLoader}`` for SQuAD v2.0 using Table 6 defaults."""

    import torch  # noqa: WPS433 (lazy)
    from torch.utils.data import DataLoader  # noqa: WPS433 (lazy)

    if batch_size is None:
        batch_size = get_squad_hparams()["batch_size"]

    loaders: Dict[str, Any] = {}
    for split in splits:
        key = canonical_split(split)
        training = key == "train"
        features, _examples, _records = build_squad_features(
            key,
            tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=training,
            data_dir=data_dir,
            hf_name=hf_name,
            use_datasets=use_datasets,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        dataset = SquadDataset(features, training=training)
        collator = SquadCollator(tokenizer=tokenizer)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loaders[key] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(training and shuffle_train),
            collate_fn=collator,
            num_workers=int(num_workers),
            generator=generator if (training and shuffle_train) else None,
            **kwargs,
        )
    return loaders


def get_squad_hparams() -> Dict[str, Any]:
    """Table 6 hyper-parameters for the SQuAD runs (falls back to a local copy)."""

    try:  # pragma: no cover - depends on sibling module
        from .glue import TABLE6  # type: ignore

        if "squad" in TABLE6:
            entry = TABLE6["squad"]
            if hasattr(entry, "as_dict"):
                return dict(entry.as_dict())
            return {
                "learning_rate": getattr(entry, "learning_rate", SQUAD_TABLE6["learning_rate"]),
                "batch_size": getattr(entry, "batch_size", SQUAD_TABLE6["batch_size"]),
                "epochs": getattr(entry, "epochs", SQUAD_TABLE6["epochs"]),
                "distill_epochs": getattr(
                    entry, "distill_epochs", SQUAD_TABLE6["distill_epochs"]
                ),
                "target_sparsity": getattr(
                    entry, "target_sparsity", SQUAD_TABLE6["target_sparsity"]
                ),
            }
    except Exception:  # noqa: BLE001 - standalone usage
        pass
    return dict(SQUAD_TABLE6)


# ---------------------------------------------------------------------------
# Post-processing (span selection) -- HF run_qa compatible
# ---------------------------------------------------------------------------


def get_best_indexes(logits: Sequence[float], n_best_size: int) -> List[int]:
    """Return the ``n_best_size`` indices with the highest logits."""

    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best = [index for index, _ in index_and_score[: max(0, int(n_best_size))]]
    while len(best) < min(n_best_size, len(logits)):
        best.append(0)
    return best


def normalize_answer(text: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between two (already normalisable) strings."""

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _v2_metric(prediction: str, ground_truths: Sequence[str]) -> Tuple[float, float]:
    """F1 / EM for one question, with SQuAD v2.0 no-answer handling."""

    pred = prediction.strip()
    if not pred:
        pred = NO_ANSWER
    golds = [g for g in ground_truths]
    if not golds:
        golds = [NO_ANSWER]

    best_f1 = best_em = float(0.0)
    pred_is_null = normalize_answer(pred) == normalize_answer(NO_ANSWER)
    for ground_truth in golds:
        gold_is_null = normalize_answer(ground_truth) == normalize_answer(NO_ANSWER)
        if pred_is_null and gold_is_null:
            em, score = 1.0, 1.0
        elif pred_is_null or gold_is_null:
            em, score = 0.0, 0.0
        else:
            if ground_truth in pred:
                # simplified but matches the official metric for contained answers
                em = 1.0 if normalize_answer(pred) == normalize_answer(ground_truth) else 0.0
            else:
                em = 1.0 if exact_match_score(pred, ground_truth) else 0.0
            if em:
                score = 1.0
            else:
                score = f1_score(pred, ground_truth)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, score)
    return best_f1, best_em


def compute_squad_metrics(
    predictions: Dict[str, str],
    references: Dict[str, Any],
    *,
    no_answer_probs: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """Official-style SQuAD v2.0 metrics.

    Args:
        predictions: ``{id: predicted_answer_string}``.  An empty string (or
            ``"no answer"``) denotes ANull (no answer).
        references: ``{id: {"answers": [str, ...], "is_impossible": bool}}`` or
            ``{id: [answers]}``.
        no_answer_probs: optional ``{id: p}`` probabilities of the no-answer
            bucket, used only to compute the THRESHOLDED ("Best") scores.

    Returns:
        dict with ``exact_match``, ``f1``, ``total``, ``total_has_ans``,
        ``total_no_ans``, ``has_ans_exact``, ``has_ans_f1``, ``no_ans_exact``,
        ``no_ans_f1``, ``best_exact``, ``best_f1``, ``best_exact_thresh``,
        ``best_f1_thresh``.
    """

    import numpy as np  # noqa: WPS433 (lazy)

    if isinstance(references, (list, tuple)):
        references = {str(i): ref for i, ref in enumerate(references)}

    ids = [key for key in predictions.keys() if key in references]

    total = len(ids)
    total_has_ans = 0
    total_no_ans = 0
    exact_scores = np.zeros(total, dtype=np.float64)
    f1_scores = np.zeros(total, dtype=np.float64)
    best_exact = np.zeros(total, dtype=np.float64)
    best_f1 = np.zeros(total, dtype=np.float64)
    best_exact_thresh = np.zeros(total, dtype=np.float64)
    best_f1_thresh = np.zeros(total, dtype=np.float64)
    has_ans_idx: List[int] = []
    no_ans_idx: List[int] = []

    for position, key in enumerate(ids):
        reference = references[key]
        if isinstance(reference, dict):
            ground_truths = list(reference.get("answers", []) or [])
            is_impossible = bool(reference.get("is_impossible", not ground_truths))
        elif isinstance(reference, (list, tuple)):
            ground_truths = list(reference)
            is_impossible = len(ground_truths) == 0
        else:  # single string reference
            ground_truths = [str(reference)]
            is_impossible = False

        if is_impossible or not ground_truths:
            if is_impossible:
                ground_truths = [NO_ANSWER]

        prediction = predictions[key]
        if isinstance(prediction, (list, tuple)):
            candidates = list(prediction)
        else:
            candidates = [prediction]

        # ---- raw (non-thresholded) score against the gold answers --------
        best_em = best_f1_raw = 0.0
        for candidate in candidates:
            em, score = _v2_metric(str(candidate), ground_truths)
            best_em = max(best_em, em)
            best_f1_raw = max(best_f1_raw, score)
        best_exact[position] = best_em
        best_f1[position] = best_f1_raw

        # ---- "Best" scores using the no-answer probability ----------------
        no_ans_p = (
            float(no_answer_probs.get(key, 0.0))
            if no_answer_probs is not None
            else None
        )
        if no_ans_p is not None:
            candidates_with_null = list(candidates) + [NO_ANSWER]
            probs = [1.0 - no_ans_p] * len(candidates) + [no_ans_p]
            order = np.argsort(probs)[::-1]
            probs_sorted = [probs[i] for i in order]
            threshs = np.concatenate([np.ones(len(probs)), np.zeros(1)])
            threshs = sorted(threshs, reverse=True)[: len(probs_sorted)]
            best_em_t = best_f1_t = 0.0
            for threshold in threshs:
                index = int(np.argmax(probs_sorted >= threshold))
                candidate = str(candidates_with_null[order[index]])
                em, score = _v2_metric(candidate, ground_truths)
                best_em_t = max(best_em_t, em)
                best_f1_t = max(best_f1_t, score)
            best_exact_thresh[position] = best_em_t
            best_f1_thresh[position] = best_f1_t
        else:
            best_exact_thresh[position] = best_em
            best_f1_thresh[position] = best_f1_raw

        # ---- aggregated metric uses ONLY the top candidate ---------------
        top = str(candidates[0]) if candidates else NO_ANSWER
        if str(top).strip() == "":
            top = NO_ANSWER
        em, score = _v2_metric(top, ground_truths)
        exact_scores[position] = em
        f1_scores[position] = score

        if is_impossible or not ground_truths or (
            len(ground_truths) == 1 and ground_truths[0] == NO_ANSWER
        ):
            total_no_ans += 1
            no_ans_idx.append(position)
        else:
            total_has_ans += 1
            has_ans_idx.append(position)

    has_ans_exact = float(exact_scores[has_ans_idx].mean()) if has_ans_idx else 0.0
    has_ans_f1 = float(f1_scores[has_ans_idx].mean()) if has_ans_idx else 0.0
    no_ans_exact = float(exact_scores[no_ans_idx].mean()) if no_ans_idx else 0.0
    no_ans_f1 = float(f1_scores[no_ans_idx].mean()) if no_ans_idx else 0.0
    best_exact_mean = float(best_exact.mean()) if total else 0.0
    best_f1_mean = float(best_f1.mean()) if total else 0.0
    best_exact_thresh_mean = float(best_exact_thresh.mean()) if total else 0.0
    best_f1_thresh_mean = float(best_f1_thresh.mean()) if total else 0.0

    return {
        "exact_match": float(exact_scores.mean()) if total else 0.0,
        "f1": float(f1_scores.mean()) if total else 0.0,
        "total": float(total),
        "total_has_ans": float(total_has_ans),
        "total_no_ans": float(total_no_ans),
        "has_ans_exact": has_ans_exact,
        "has_ans_f1": has_ans_f1,
        "no_ans_exact": no_ans_exact,
        "no_ans_f1": no_ans_f1,
        "best_exact": best_exact_mean,
        "best_f1": best_f1_mean,
        "best_exact_thresh": best_exact_thresh_mean,
        "best_f1_thresh": best_f1_thresh_mean,
    }


def squad_metric_summary(metrics: Dict[str, float]) -> str:
    """Human-readable one-line summary (dev F1 is the reported Table 2 metric)."""

    return (
        "SQuAD v2.0 | EM {em:.2f} | F1 {f1:.2f} | best_EM {bem:.2f} | "
        "best_F1 {bf1:.2f} | total {n:.0f} (has-ans {ha:.0f}, no-ans {na:.0f})"
    ).format(
        em=100 * metrics.get("exact_match", 0.0),
        f1=100 * metrics.get("f1", 0.0),
        bem=100 * metrics.get("best_exact", 0.0),
        bf1=100 * metrics.get("best_f1", 0.0),
        n=metrics.get("total", 0.0),
        ha=metrics.get("total_has_ans", 0.0),
        na=metrics.get("total_no_ans", 0.0),
    )


def _token_ids_to_text(
    input_ids: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    context: str,
    start: int,
    end: int,
    tokenizer: Any = None,
) -> str:
    """Recover the answer string for tokens ``[start, end]``."""

    start = max(0, int(start))
    end = min(len(input_ids) - 1, int(end))
    if end < start:
        return ""
    if offsets and len(offsets) > end:
        char_start = offsets[start][0]
        char_end = offsets[end][1]
        if char_end > char_start and char_start >= 0:
            return context[char_start:char_end].strip()
    if tokenizer is not None:
        tokens = tokenizer.convert_ids_to_tokens(list(input_ids[start : end + 1]))
        tokens = [t for t in tokens if not (isinstance(t, str) and t.startswith("<"))]
        try:
            return tokenizer.convert_tokens_to_string(tokens).strip()
        except Exception:  # noqa: BLE001
            return " ".join(tokens).strip()
    return ""


def write_predictions(
    features: Sequence[SquadFeatures],
    examples: Sequence[SquadExample],
    all_results: Sequence[Tuple[int, Sequence[float], Sequence[float]]],
    *,
    n_best_size: int = SQUAD_N_BEST_SIZE,
    max_answer_length: int = SQUAD_MAX_ANSWER_LENGTH,
    version_2_with_negative: bool = True,
    null_score_diff_threshold: float = SQUAD_NULL_SCORE_DIFF_THRESHOLD,
    tokenizer: Any = None,
    output_prediction_file: Optional[str] = None,
    output_nbest_file: Optional[str] = None,
    output_null_log_odds_file: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    """Convert ``(start_logits, end_logits)`` into answer strings.

    Returns ``(predictions, nbest, scores_diff)`` where ``predictions`` maps
    example id -> answer text (ANull becomes ``""``).
    """

    import numpy as np  # noqa: WPS433 (lazy)

    results_by_id = {
        int(unique_id): (start_logits, end_logits)
        for unique_id, start_logits, end_logits in all_results
    }
    example_by_index = {index: example for index, example in enumerate(examples)}

    preliminary: Dict[int, Tuple[float, SquadPrediction]] = {}
    for feature in features:
        if feature.unique_id not in results_by_id:
            continue
        start_logits, end_logits = results_by_id[feature.unique_id]
        start_logits = [float(x) for x in start_logits]
        end_logits = [float(x) for x in end_logits]
        if not start_logits or not end_logits:
            continue
        if len(start_logits) != len(end_logits):  # defensive alignment
            length = min(len(start_logits), len(end_logits))
            start_logits, end_logits = start_logits[:length], end_logits[:length]

        start_indexes = get_best_indexes(start_logits, n_best_size)
        end_indexes = get_best_indexes(end_logits, n_best_size)

        example = example_by_index.get(feature.example_index)
        context = example.context_text if example is not None else ""

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(feature.p_mask) or end_index >= len(feature.p_mask):
                    continue
                if feature.p_mask[start_index] > 0 or feature.p_mask[end_index] > 0:
                    continue  # part of the question
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > max_answer_length:
                    continue
                if (
                    len(feature.token_is_max_context) > 0
                    and not feature.token_is_max_context.get(start_index, True)
                ):
                    continue
                text = _token_ids_to_text(
                    feature.input_ids,
                    feature.offset_mapping,
                    context,
                    start_index,
                    end_index,
                    tokenizer=tokenizer,
                )
                score = start_logits[start_index] + end_logits[end_index]
                if feature.unique_id in preliminary:
                    best_score, _ = preliminary[feature.unique_id]
                    if score <= best_score:
                        continue
                preliminary[feature.unique_id] = (
                    score,
                    SquadPrediction(
                        unique_id=feature.unique_id,
                        example_index=feature.example_index,
                        span_index=feature.span_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index],
                        score=score,
                        text=text,
                    ),
                )

    predictions: Dict[str, str] = {}
    nbest: Dict[str, List[Dict[str, Any]]] = {}
    scores_diff: Dict[str, float] = {}

    features_by_example: Dict[int, List[SquadFeatures]] = collections.defaultdict(list)
    for feature in features:
        features_by_example[feature.example_index].append(feature)

    for example_index, example in example_by_index.items():
        candidates: List[SquadPrediction] = []
        null_score = None
        for feature in features_by_example.get(example_index, []):
            entry = preliminary.get(feature.unique_id)
            if entry is None:
                continue
            if null_score is None or (
                -1e4 if null_score is None else null_score
            ) < feature.cls_index:
                pass
            start_logits, end_logits = results_by_id.get(feature.unique_id, (None, None))
            if start_logits is not None and end_logits is not None:
                cls_logit = float(start_logits[feature.cls_index]) + float(
                    end_logits[feature.cls_index]
                )
                null_score = cls_logit if null_score is None else max(null_score, cls_logit)
            candidates.append(entry[1])

        candidates.sort(key=lambda p: p.score, reverse=True)
        nbest[example.qas_id] = [
            {
                "text": p.text,
                "start_logit": p.start_logit,
                "end_logit": p.end_logit,
                "score": p.score,
                "start_index": p.start_index,
                "end_index": p.end_index,
            }
            for p in candidates[:n_best_size]
        ]

        if not candidates:
            predictions[example.qas_id] = ""
            if null_score is not None:
                scores_diff[example.qas_id] = null_score
            continue

        best = candidates[0]
        if not version_2_with_negative or null_score is None:
            predictions[example.qas_id] = best.text
        else:
            scores_diff[example.qas_id] = null_score - best.score
            if (null_score - best.score) > null_score_diff_threshold:
                predictions[example.qas_id] = ""
            else:
                predictions[example.qas_id] = best.text

    if output_prediction_file:
        with open(output_prediction_file, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2, sort_keys=True)
    if output_nbest_file:
        with open(output_nbest_file, "w", encoding="utf-8") as handle:
            json.dump(nbest, handle, indent=2, sort_keys=True)
    if output_null_log_odds_file:
        with open(output_null_log_odds_file, "w", encoding="utf-8") as handle:
            json.dump(scores_diff, handle, indent=2, sort_keys=True)

    return predictions, nbest, scores_diff


def compute_predictions_logits(
    features: Sequence[SquadFeatures],
    examples: Sequence[SquadExample],
    all_results: Sequence[Tuple[int, Sequence[float], Sequence[float]]],
    **kwargs: Any,
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]], Dict[str, float]]:
    """Alias of :func:`write_predictions` (HF naming)."""

    return write_predictions(features, examples, all_results, **kwargs)


def references_for_examples(
    examples: Sequence[SquadExample],
) -> Dict[str, Dict[str, Any]]:
    """Build the ``references`` dict consumed by :func:`compute_squad_metrics`."""

    return {
        example.qas_id: {
            "answers": [] if example.is_impossible else list(example.answers),
            "is_impossible": bool(example.is_impossible or not example.answers),
        }
        for example in examples
    }


def evaluate_predictions(
    predictions: Dict[str, str],
    examples: Sequence[SquadExample],
    *,
    no_answer_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Convenience wrapper: examples -> metrics dict (dev F1 in ``"f1"``)."""

    return compute_squad_metrics(
        predictions,
        references_for_examples(examples),
        no_answer_probs=no_answer_probs,
    )


def span_logits_to_answers(
    model_outputs: Any,
    features: Sequence[SquadFeatures],
    examples: Sequence[SquadExample],
    tokenizer: Any = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """End-to-end helper: model logits -> SQuAD v2.0 metrics dict."""

    start_logits = getattr(model_outputs, "start_logits", None)
    end_logits = getattr(model_outputs, "end_logits", None)
    if start_logits is None or end_logits is None:
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) >= 2:
            start_logits, end_logits = model_outputs[0], model_outputs[1]
        else:
            raise ValueError("model_outputs must provide start_logits / end_logits")

    all_results = [
        (feature.unique_id, start_logits[i].tolist(), end_logits[i].tolist())
        for i, feature in enumerate(features)
    ]
    predictions, _nbest, scores_diff = write_predictions(
        features, examples, all_results, tokenizer=tokenizer, **kwargs
    )
    no_answer_probs = None
    if scores_diff:
        # map score diff -> coarse no-answer probability (monotone)
        if len(scores_diff) > 0:
            max_abs = max(abs(v) for v in scores_diff.values()) or 1.0
            no_answer_probs = {
                key: float(1.0 / (1.0 + pow(2.718281828, -value / max_abs)))
                for key, value in scores_diff.items()
            }
    return evaluate_predictions(predictions, examples, no_answer_probs=no_answer_probs)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Lightweight sanity checks that need no downloads/GPU."""

    # --- metrics -----------------------------------------------------------
    refs = {
        "1": {"answers": ["the quick brown fox"], "is_impossible": False},
        "2": {"answers": [], "is_impossible": True},
    }
    preds = {"1": "quick brown fox", "2": ""}
    metrics = compute_squad_metrics(preds, refs)
    assert 0.0 <= metrics["f1"] <= 1.0, metrics
    assert metrics["total"] == 2, metrics
    assert metrics["total_has_ans"] == 1 and metrics["total_no_ans"] == 1, metrics
    assert metrics["has_ans_f1"] > 0.5, metrics
    assert metrics["no_ans_f1"] == 1.0, metrics

    # perfect predictions -> F1 1.0
    perfect = {"1": "the quick brown fox", "2": "no answer"}
    perfect_metrics = compute_squad_metrics(perfect, refs)
    assert abs(perfect_metrics["f1"] - 1.0) < 1e-8, perfect_metrics
    assert abs(perfect_metrics["exact_match"] - 1.0) < 1e-8, perfect_metrics

    # --- normalisation / span helpers -------------------------------------
    assert normalize_answer("The  Fox!") == "fox"
    assert get_best_indexes([0.1, 0.9, 0.5], 2) == [1, 2]
    spans = _doc_spans(10, 4, 2)
    assert spans and spans[0][0] == 0, spans
    assert _doc_spans(3, 4, 2)[0][1] == 3

    # --- examples ----------------------------------------------------------
    records = [
        {
            "id": "q1",
            "question": "Who?",
            "context": "Alice went home.",
            "answers": {"text": ["Alice"], "answer_start": [0]},
        },
        {
            "id": "q2",
            "question": "Where?",
            "context": "Alice went home.",
            "answers": {"text": [], "answer_start": []},
            "is_impossible": True,
        },
    ]
    examples = squad_examples_from_records(records, is_training=True)
    assert len(examples) == 2, examples
    assert examples[0].answer_text == "Alice"
    assert examples[1].is_impossible

    # --- raw json flattening ----------------------------------------------
    raw = {
        "data": [
            {
                "title": "t",
                "paragraphs": [
                    {
                        "context": "Alice went home.",
                        "qas": [
                            {
                                "id": "q1",
                                "question": "Who?",
                                "answers": [{"text": "Alice", "answer_start": 0}],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    flat = squad_records_from_raw(raw)
    assert len(flat) == 1 and flat[0]["answers"]["text"] == ["Alice"], flat

    # --- Table 6 hparams ---------------------------------------------------
    hp = get_squad_hparams()
    assert hp["batch_size"] == 32 and hp["epochs"] == 40 and hp["distill_epochs"] == 20, hp
    assert abs(hp["learning_rate"] - 2e-4) < 1e-12, hp
    return True


if __name__ == "__main__":  # pragma: no cover
    _self_test()
    print("squad.py self-test passed")
