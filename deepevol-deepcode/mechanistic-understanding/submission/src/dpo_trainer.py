"""Direct Preference Optimization (DPO) trainer for the toxicity-alignment study.

This module implements Section 4.1 (Eq. 1) and Section 4.2 of

    "A Mechanistic Understanding of Alignment Algorithms:
     A Case Study on DPO and Toxicity"

The DPO objective is

    L_DPO = -E[ log sigma( beta * log P - beta * log N ) ]

    P = pi_theta(y_+ | w) / pi_ref(y_+ | w)
    N = pi_theta(y_- | w) / pi_ref(y_- | w)

where ``y_+`` is the preferred (non-toxic) continuation, ``y_-`` is the
non-preferred (toxic) continuation, ``pi_ref`` is the frozen original language
model (GPT2) and ``pi_theta`` are the weights being updated (GPT2_DPO).

Data-wise, the preference pairs come from :mod:`data.pairwise` (Section 4.2:
24,576 Wikitext-2 prompt pairs, positive = greedy GPT2 continuation, negative =
PPLM toxic continuation).  Training hyperparameters follow Appendix E, Table 8:

    learning rate              1e-6
    batch size                 4
    optimizer                  RMSProp
    gradient accumulation      1
    max gradient norm          10
    validation metric          loss/valid
    validation patience        10
    DPO beta                   0.1

The paper trains "until validation loss converges with a patience value of 10,
which occurs after approximately 6,700 sample pairs".

All heavy imports (``torch``, ``datasets``) are performed lazily inside
functions so importing this module stays cheap.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

# --------------------------------------------------------------------------------------
# Constants (paper: Section 4.1, 4.2, Appendix E Table 8)
# --------------------------------------------------------------------------------------

DPO_BETA = 0.1
DPO_LR = 1e-6
DPO_BATCH_SIZE = 4
DPO_GRAD_ACCUM = 1
DPO_MAX_GRAD_NORM = 10.0
DPO_OPTIMIZER = "rmsprop"
DPO_VALIDATION_METRIC = "loss/valid"
DPO_VALIDATION_PATIENCE = 10
N_PAIRS = 24_576
CONVERGENCE_EXAMPLES = 6_700

GPT2_MEDIUM = "openai-community/gpt2-medium"
DEFAULT_OUTPUT_DIR = os.path.join("artifacts", "models", "gpt2_dpo")
DEFAULT_PAIRS_PATH = os.path.join("artifacts", "data", "pairs.jsonl")
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
TRAINER_STATE_FILENAME = "trainer_state.json"

IGNORE_INDEX = -100


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class DPOConfig:
    """Hyperparameters and bookkeeping for DPO training (Appendix E, Table 8)."""

    # --- paper hyperparameters ---------------------------------------------------------
    beta: float = DPO_BETA
    learning_rate: float = DPO_LR
    batch_size: int = DPO_BATCH_SIZE
    grad_accum: int = DPO_GRAD_ACCUM
    max_grad_norm: float = DPO_MAX_GRAD_NORM
    optimizer: str = DPO_OPTIMIZER
    validation_metric: str = DPO_VALIDATION_METRIC
    patience: int = DPO_VALIDATION_PATIENCE

    # --- unspecified in the paper: sensible defaults ------------------------------------
    num_epochs: int = 3
    weight_decay: float = 0.0
    momentum: float = 0.0
    warmup_steps: int = 0
    lr_scheduler: str = "none"  # "none" | "linear" | "cosine"
    max_length: int = 128
    eval_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 0  # 0 -> only save best / final
    seed: int = 0
    device: Optional[str] = None
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    cache_reference_logps: bool = True
    early_stopping: bool = True
    max_steps: int = -1
    max_train_examples: Optional[int] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    model_name: str = GPT2_MEDIUM
    reference_model_name: Optional[str] = None  # defaults to model_name
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # paper uses the sigmoid (Eq. 1) form
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DPOConfig":
        """Build a config from a (possibly partial / nested) dict."""
        flat: Dict[str, Any] = {}
        if not isinstance(data, dict):
            return cls()
        # support {"dpo": {...}} / {"train": {...}} / {"training": {...}} nesting
        merged: Dict[str, Any] = {}
        for key in ("dpo", "train", "training", "dpo_train", "train_dpo"):
            section = data.get(key)
            if isinstance(section, dict):
                merged.update(section)
        merged.update({k: v for k, v in data.items() if not isinstance(v, dict)})
        if isinstance(data.get("model"), dict):
            merged.setdefault("model_name", data["model"].get("name"))
        elif isinstance(data.get("model"), str):
            merged.setdefault("model_name", data["model"])

        aliases = {
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "batch": "batch_size",
            "gradient_accumulation_steps": "grad_accum",
            "gradient_accumulation": "grad_accum",
            "max_grad_norm": "max_grad_norm",
            "max_gradient_norm": "max_grad_norm",
            "dpo_beta": "beta",
            "validation_patience": "patience",
            "epochs": "num_epochs",
            "model": "model_name",
            "model_name": "model_name",
            "out_dir": "output_dir",
        }
        valid = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        for key, value in merged.items():
            if value is None:
                continue
            target = aliases.get(key, key)
            if target in valid:
                flat[target] = value
        return cls(**flat)


# --------------------------------------------------------------------------------------
# Tokenisation / dataset
# --------------------------------------------------------------------------------------


def _ensure_pad_token(tokenizer) -> None:
    """GPT2 has no pad token; reuse EOS so batches can be padded."""
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:  # pragma: no cover - defensive
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _as_pair_dicts(pairs: Any) -> List[Dict[str, str]]:
    """Normalise many pair containers into ``[{'prompt','chosen','rejected'}]``.

    Accepts:
      * ``PairwiseDataset`` / ``PairSplit`` (``data.pairwise``)
      * iterables of ``PairExample``
      * HF-style dicts with ``prompt``/``chosen``/``rejected``
    """
    if pairs is None:
        return []
    # data.pairwise containers
    if hasattr(pairs, "to_hf"):
        try:
            return list(pairs.to_hf())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(pairs, "pairs"):
        return _as_pair_dicts(pairs.pairs)
    out: List[Dict[str, str]] = []
    for item in pairs:
        if hasattr(item, "to_hf"):
            out.append(dict(item.to_hf()))
        elif isinstance(item, dict):
            prompt = item.get("prompt", item.get("w", ""))
            chosen = item.get("chosen", item.get("preferred", item.get("y_plus", "")))
            rejected = item.get(
                "rejected", item.get("non_preferred", item.get("y_minus", ""))
            )
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported pair type: {type(item)!r}")
    return out


def encode_pair(tokenizer, prompt: str, continuation: str, max_length: int = 128):
    """Tokenise ``prompt + continuation`` and mask the prompt tokens.

    Returns ``(input_ids, attention_mask, labels)`` as plain python lists, where
    ``labels`` is ``-100`` over the prompt (and padding) so that only the
    continuation contributes to the DPO log-probabilities.
    """
    prompt = "" if prompt is None else str(prompt)
    continuation = "" if continuation is None else str(continuation)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    # Defensive: tokenisation of the concatenation may differ from the sum when
    # the continuation starts mid-word; fall back to concatenating the parts.
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if full_ids != list(prompt_ids) + list(cont_ids):
        full_ids = list(prompt_ids) + list(cont_ids)

    if len(full_ids) > max_length:
        # keep the tail of the prompt and the beginning of the continuation
        # by truncating the prompt first (mirrors common preference-data practice)
        n_cont = min(len(cont_ids), max_length - 1)
        n_prompt = max(0, max_length - n_cont)
        prompt_ids = prompt_ids[-n_prompt:] if n_prompt else []
        full_ids = prompt_ids + cont_ids[:n_cont]

    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * n_prompt + full_ids[n_prompt:]
    attention_mask = [1] * len(full_ids)
    return full_ids, attention_mask, labels


class DPODataset:
    """Torch-compatible dataset of tokenised preference pairs.

    Parameters
    ----------
    pairs:
        Any container accepted by :func:`_as_pair_dicts`.
    tokenizer:
        HuggingFace tokenizer (GPT2 byte-level BPE).
    max_length:
        Maximum number of tokens (prompt + continuation).
    """

    def __init__(self, pairs: Any, tokenizer, max_length: int = 128):
        self.pairs = _as_pair_dicts(pairs)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        _ensure_pad_token(tokenizer)
        self._cache: Optional[List[Tuple[List[int], List[int], List[int]]]] = None

    # -- encoding -------------------------------------------------------------------
    def _encode_all(self) -> List[Tuple[List[int], List[int], List[int]]]:
        if self._cache is None:
            self._cache = [
                encode_pair(
                    self.tokenizer,
                    p["prompt"],
                    p.get("chosen", ""),
                    self.max_length,
                )
                for p in self.pairs
            ] + [
                encode_pair(
                    self.tokenizer,
                    p["prompt"],
                    p.get("rejected", ""),
                    self.max_length,
                )
                for p in self.pairs
            ]
        return self._cache

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.pairs[idx]
        c_ids, c_mask, c_lab = encode_pair(
            self.tokenizer, row["prompt"], row.get("chosen", ""), self.max_length
        )
        r_ids, r_mask, r_lab = encode_pair(
            self.tokenizer, row["prompt"], row.get("rejected", ""), self.max_length
        )
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels": c_lab,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels": r_lab,
            "index": idx,
        }

    # -- collation ------------------------------------------------------------------
    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch  # lazy

        pad_id = self.tokenizer.pad_token_id

        def _stack(key_ids: str, key_mask: str, key_lab: str):
            seqs = [b[key_ids] for b in batch]
            max_len = max(len(s) for s in seqs)
            ids, mask, labels = [], [], []
            for b in batch:
                s = b[key_ids]
                pad = max_len - len(s)
                ids.append(s + [pad_id] * pad)
                mask.append(b[key_mask] + [0] * pad)
                labels.append(b[key_lab] + [IGNORE_INDEX] * pad)
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )

        c_ids, c_mask, c_lab = _stack(
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels"
        )
        r_ids, r_mask, r_lab = _stack(
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
        )
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels": c_lab,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels": r_lab,
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        }

    # -- helpers --------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        n_tokens = 0
        n_pairs = len(self.pairs)
        for row in self.pairs:
            for key in ("chosen", "rejected"):
                _, _, labels = encode_pair(
                    self.tokenizer, row["prompt"], row.get(key, ""), self.max_length
                )
                n_tokens += sum(1 for x in labels if x != IGNORE_INDEX)
        return {"n_pairs": n_pairs, "n_continuation_tokens": n_tokens}


def make_loader(dataset: DPODataset, batch_size: int, shuffle: bool, seed: int = 0):
    """Build a ``torch.utils.data.DataLoader`` with the dataset's collate_fn."""
    import torch  # lazy

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        collate_fn=dataset.collate,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


# --------------------------------------------------------------------------------------
# Core maths (Eq. 1)
# --------------------------------------------------------------------------------------


def sequence_logprobs(
    model,
    input_ids,
    attention_mask=None,
    labels=None,
    average_log_prob: bool = False,
):
    """Sum of log-probabilities of the completion tokens (``log pi(y | w)``).

    ``labels`` must contain ``-100`` for prompt/padding positions.  Returns a
    1-D tensor of per-sequence log-probabilities.
    """
    import torch  # lazy

    if labels is None:
        raise ValueError("labels are required (use -100 to mask the prompt)")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    total = gathered.sum(dim=-1)
    if average_log_prob:
        denom = mask.sum(dim=-1).clamp(min=1)
        return total / denom, mask.sum(dim=-1)
    return total


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float = DPO_BETA,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
):
    """Equation 1 of the paper.

    ``P = pi_theta(y+|w) / pi_ref(y+|w)``, ``N = pi_theta(y-|w) / pi_ref(y-|w)``
    and ``L = -E[log sigma(beta log P - beta log N)]``.

    Returns ``(loss, chosen_rewards, rejected_rewards)`` where the implicit
    rewards are ``beta * log(pi_theta / pi_ref)`` per sequence.
    """
    import torch  # lazy

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios  # log P - log N

    if loss_type == "sigmoid":
        if label_smoothing > 0.0:
            losses = (
                -torch.nn.functional.logsigmoid(beta * logits) * (1.0 - label_smoothing)
                - torch.nn.functional.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            losses = -torch.nn.functional.logsigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = torch.relu(1.0 - beta * logits)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def dpo_accuracy(chosen_rewards, rejected_rewards):
    """Fraction of pairs where the implicit reward prefers the non-toxic sample."""
    import torch  # lazy

    return (chosen_rewards > rejected_rewards).float().mean()


def implicit_reward_margin(policy_logps, reference_logps, beta: float = DPO_BETA):
    """``beta * log(pi_theta / pi_ref)`` for one sample (implicit reward)."""
    return beta * (policy_logps - reference_logps)


def log_ratio(policy_logps, reference_logps) -> "object":
    """``log(pi_theta(y|w)) - log(pi_ref(y|w))`` (used by the analyses)."""
    return policy_logps - reference_logps


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------


@dataclass
class DPOTrainResult:
    """Container for everything produced by one DPO training run."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    model_name: str = GPT2_MEDIUM
    n_train_pairs: int = 0
    n_valid_pairs: int = 0
    steps: int = 0
    epochs_run: float = 0.0
    train_examples_seen: int = 0
    best_valid_loss: float = float("nan")
    best_step: int = -1
    final_train_loss: float = float("nan")
    final_valid_loss: float = float("nan")
    final_valid_accuracy: float = float("nan")
    history: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "n_train_pairs": self.n_train_pairs,
            "n_valid_pairs": self.n_valid_pairs,
            "steps": self.steps,
            "epochs_run": self.epochs_run,
            "train_examples_seen": self.train_examples_seen,
            "best_valid_loss": self.best_valid_loss,
            "best_step": self.best_step,
            "final_train_loss": self.final_train_loss,
            "final_valid_loss": self.final_valid_loss,
            "final_valid_accuracy": self.final_valid_accuracy,
            "converged_examples": CONVERGENCE_EXAMPLES,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary()
        return _json_safe(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DPOTrainResult":
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars into JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    for attr in ("item", "tolist"):
        if hasattr(obj, attr) and not isinstance(obj, type):
            try:
                return _json_safe(getattr(obj, attr)())
            except Exception:  # pragma: no cover - defensive
                pass
    try:
        import numpy as np  # lazy

        if isinstance(obj, np.generic):
            return _json_safe(obj.item())
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    except Exception:  # pragma: no cover
        pass
    return str(obj)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class DPOTrainer:
    """DPO trainer implementing Eq. 1 with the Table 8 hyperparameters.

    Parameters
    ----------
    model:
        Policy model (GPT2 / GPT2-medium), will be updated in place.
    ref_model:
        Frozen reference model.  If ``None``, a copy of ``model`` is made.
    tokenizer:
        Tokenizer used for encoding pairs.
    config:
        :class:`DPOConfig` instance (or plain dict).
    train_pairs / valid_pairs:
        Any container accepted by :func:`_as_pair_dicts`.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: Optional[DPOConfig] = None,
        train_pairs: Any = None,
        valid_pairs: Any = None,
    ):
        import torch  # lazy

        self.config = config if isinstance(config, DPOConfig) else DPOConfig.from_dict(config or {})
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._resolve_device(self.config.device)

        _ensure_pad_token(tokenizer)

        # Frozen reference model: original (pre-DPO) weights, pi_ref.
        if ref_model is None:
            import copy

            ref_model = copy.deepcopy(model)
        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.model.to(self.device)
        self.ref_model.to(self.device)
        if self.config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:  # pragma: no cover - optional
                pass

        # Datasets / loaders ---------------------------------------------------------
        self.train_dataset = (
            DPODataset(train_pairs, tokenizer, self.config.max_length)
            if train_pairs is not None
            else None
        )
        self.valid_dataset = (
            DPODataset(valid_pairs, tokenizer, self.config.max_length)
            if valid_pairs is not None
            else None
        )
        self.train_loader = (
            make_loader(
                self.train_dataset,
                self.config.batch_size,
                shuffle=True,
                seed=self.config.seed,
            )
            if self.train_dataset is not None and len(self.train_dataset) > 0
            else None
        )
        self.valid_loader = (
            make_loader(
                self.valid_dataset,
                max(1, self.config.batch_size),
                shuffle=False,
                seed=self.config.seed,
            )
            if self.valid_dataset is not None and len(self.valid_dataset) > 0
            else None
        )

        # Optimizer (Table 8: RMSProp) -------------------------------------------------
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=self._use_amp())
            if hasattr(torch, "cuda") and torch.cuda.is_available()
            else None
        )

        # Reference log-prob cache (values of log pi_ref, frozen) ----------------------
        self._ref_cache: Dict[Tuple[str, int], "object"] = {}
        self._precomputed = False

        # bookkeeping
        self.global_step = 0
        self.examples_seen = 0
        self.best_valid_loss = float("inf")
        self.best_step = -1
        self.evals_without_improvement = 0
        self.history: List[Dict[str, Any]] = []
        self._stop = False
        self._best_state: Optional[Dict[str, Any]] = None

    # -- setup helpers ---------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: Optional[str]):
        import torch  # lazy

        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _use_amp(self) -> bool:
        import torch  # lazy

        return bool(
            self.config.mixed_precision
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

    def _build_optimizer(self):
        import torch  # lazy

        params = [p for p in self.model.parameters() if p.requires_grad]
        name = (self.config.optimizer or "rmsprop").lower()
        if name in ("rmsprop", "rms_prop"):
            return torch.optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
            )
        if name in ("adamw", "adam_w"):
            return torch.optim.AdamW(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        if name == "adam":
            return torch.optim.Adam(params, lr=self.config.learning_rate)
        if name == "sgd":
            return torch.optim.SGD(
                params, lr=self.config.learning_rate, momentum=self.config.momentum
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer!r}")

    def _build_scheduler(self):
        import torch  # lazy

        if self.train_loader is None or self.config.lr_scheduler in (None, "none"):
            return None
        steps = self._planned_steps()
        warmup = max(0, int(self.config.warmup_steps))
        if self.config.lr_scheduler == "linear":

            def fn(step: int) -> float:
                if warmup and step < warmup:
                    return (step + 1) / max(1, warmup)
                return max(0.0, (steps - step) / max(1, steps - warmup))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, fn)
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, steps))
        return None

    def _planned_steps(self) -> int:
        if self.train_loader is None:
            return 0
        per_epoch = math.ceil(len(self.train_loader) / max(1, self.config.grad_accum))
        total = per_epoch * max(1, self.config.num_epochs)
        if self.config.max_steps and self.config.max_steps > 0:
            total = min(total, int(self.config.max_steps))
        return max(1, total)

    # -- reference log-probabilities -------------------------------------------------
    def _reference_logps(self, split: str, batch: Dict[str, Any], prefix: str, label_key: str):
        """``log pi_ref(y | w)`` for a batch, with an optional cache."""
        indices = batch["index"].tolist()
        ids = batch[f"{prefix}_input_ids"].to(self.device)
        mask = batch[f"{prefix}_attention_mask"].to(self.device)
        labels = batch[label_key].to(self.device)

        if not self.config.cache_reference_logps:
            return self._forward_logps(self.ref_model, ids, mask, labels)

        import torch  # lazy

        missing = [i for i in indices if (split, i, prefix) not in self._ref_cache]
        if missing:
            positions = [indices.index(i) for i in missing]
            with torch.no_grad():
                logps = self._forward_logps(
                    self.ref_model,
                    ids[positions],
                    mask[positions],
                    labels[positions],
                )
            for pos, idx in zip(positions, missing):
                self._ref_cache[(split, idx, prefix)] = float(logps[pos].detach().cpu())
        return torch.tensor(
            [self._ref_cache[(split, i, prefix)] for i in indices],
            dtype=torch.float32,
            device=self.device,
        )

    def precompute_reference_logps(self, split: str = "train", verbose: bool = True):
        """Cache ``log pi_ref`` for a whole split (frozen reference model)."""
        loader = self.train_loader if split == "train" else self.valid_loader
        if loader is None:
            return 0
        import torch  # lazy

        total = 0
        iterator = loader
        if verbose:
            iterator = _progress(loader, f"reference logps [{split}]")
        for batch in iterator:
            for prefix, label_key in (
                ("chosen", "chosen_labels"),
                ("rejected", "rejected_labels"),
            ):
                self._reference_logps(split, batch, prefix, label_key)
            total += len(batch["index"])
        self._precomputed = True
        return total

    # -- model forward ---------------------------------------------------------------
    def _forward_logps(self, model, input_ids, attention_mask, labels):
        """``log pi(y | w)`` summed over the continuation tokens."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(IGNORE_INDEX)
        safe_labels = shift_labels.masked_fill(~mask, 0)

        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (gathered * mask).sum(dim=-1)

    # -- one optimisation step -------------------------------------------------------
    def compute_batch(self, batch: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
        """Compute the DPO loss/metrics for one batch (no backward)."""
        import torch  # lazy

        ids_c = batch["chosen_input_ids"].to(self.device)
        mask_c = batch["chosen_attention_mask"].to(self.device)
        lab_c = batch["chosen_labels"].to(self.device)
        ids_r = batch["rejected_input_ids"].to(self.device)
        mask_r = batch["rejected_attention_mask"].to(self.device)
        lab_r = batch["rejected_labels"].to(self.device)

        policy_chosen = self._forward_logps(self.model, ids_c, mask_c, lab_c)
        policy_rejected = self._forward_logps(self.model, ids_r, mask_r, lab_r)
        ref_chosen = self._reference_logps(split, batch, "chosen", "chosen_labels")
        ref_rejected = self._reference_logps(split, batch, "rejected", "rejected_labels")

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
            loss_type=self.config.loss_type,
        )
        with torch.no_grad():
            accuracy = dpo_accuracy(chosen_rewards, rejected_rewards)
            margin = (chosen_rewards - rejected_rewards).mean()
            log_ratio_mean = (
                (policy_chosen - policy_rejected).detach().mean()
                - (ref_chosen - ref_rejected).detach().mean()
            )
        return {
            "loss": loss,
            "accuracy": accuracy,
            "reward_margin": margin,
            "log_ratio_margin": log_ratio_mean,
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "n_examples": int(ids_c.shape[0]),
        }

    # -- evaluation ------------------------------------------------------------------
    def evaluate(self, split: str = "valid", verbose: bool = False) -> Dict[str, float]:
        """Mean DPO loss on a split -> reported as ``loss/valid``."""
        import torch  # lazy

        loader = self.valid_loader if split == "valid" else self.train_loader
        if loader is None:
            return {"loss": float("nan"), "accuracy": float("nan"), "n_examples": 0.0}

        was_training = self.model.training
        self.model.eval()
        totals = {"loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0}
        n = 0
        iterator = _progress(loader, f"eval[{split}]") if verbose else loader
        with torch.no_grad():
            for batch in iterator:
                out = self.compute_batch(batch, split=split)
                k = out["n_examples"]
                totals["loss"] += float(out["loss"]) * k
                totals["accuracy"] += float(out["accuracy"]) * k
                totals["reward_margin"] += float(out["reward_margin"]) * k
                n += k
        if was_training:
            self.model.train()
        n = max(1, n)
        return {
            "loss": totals["loss"] / n,
            "accuracy": totals["accuracy"] / n,
            "reward_margin": totals["reward_margin"] / n,
            "n_examples": float(n),
        }

    # -- training loop ---------------------------------------------------------------
    def train(
        self,
        num_epochs: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> DPOTrainResult:
        """Run DPO training (RMSProp, beta=0.1, patience=10 on ``loss/valid``)."""
        import torch  # lazy

        if self.train_loader is None:
            raise RuntimeError("No training pairs were provided to DPOTrainer")

        cfg = self.config
        epochs = int(num_epochs if num_epochs is not None else cfg.num_epochs)
        self.model.train()

        if cfg.cache_reference_logps and not self._precomputed:
            # precompute once for both splits (reference model is frozen)
            try:
                self.precompute_reference_logps("train", verbose=cfg.verbose)
                self.precompute_reference_logps("valid", verbose=False)
            except Exception:  # pragma: no cover - fall back to lazy caching
                pass

        start_time = _now()
        for epoch in range(epochs):
            if self._stop:
                break
            epoch_loss = 0.0
            epoch_examples = 0
            self.optimizer.zero_grad(set_to_none=True)
            iterator = (
                _progress(self.train_loader, f"epoch {epoch + 1}/{epochs}")
                if cfg.verbose
                else self.train_loader
            )
            accum = max(1, int(cfg.grad_accum))
            for step_in_epoch, batch in enumerate(iterator, start=1):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self._use_amp(),
                ):
                    out = self.compute_batch(batch, split="train")
                    loss = out["loss"] / accum

                if self.scaler is not None and self._use_amp():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step_in_epoch % accum == 0 or step_in_epoch == len(self.train_loader):
                    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                        if self.scaler is not None and self._use_amp():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg.max_grad_norm,
                        )
                    if self.scaler is not None and self._use_amp():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    epoch_loss += float(out["loss"].detach()) * out["n_examples"]
                    epoch_examples += out["n_examples"]
                    self.examples_seen += out["n_examples"]

                    if cfg.max_train_examples and self.examples_seen >= cfg.max_train_examples:
                        self._stop = True

                    if cfg.logging_steps and self.global_step % cfg.logging_steps == 0:
                        record = {
                            "step": self.global_step,
                            "epoch": epoch + float(step_in_epoch) / max(1, len(self.train_loader)),
                            "loss": float(out["loss"].detach()),
                            "accuracy": float(out["accuracy"].detach()),
                            "reward_margin": float(out["reward_margin"].detach()),
                            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                            "examples_seen": self.examples_seen,
                            "elapsed_sec": _now() - start_time,
                            "split": "train",
                        }
                        self.history.append(record)
                        if callback is not None:
                            callback(record)

                    if cfg.eval_steps and self.global_step % cfg.eval_steps == 0:
                        self._validate(epoch, callback)

                    if cfg.save_steps and self.global_step % cfg.save_steps == 0:
                        self.save(os.path.join(cfg.output_dir, f"checkpoint-{self.global_step}"))

                    if self._stop and cfg.max_train_examples:
                        break

            if epoch_examples and not cfg.logging_steps:
                self.history.append(
                    {
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "loss": epoch_loss / max(1, epoch_examples),
                        "examples_seen": self.examples_seen,
                        "split": "train",
                    }
                )

            # end-of-epoch validation so short runs still evaluate
            if not cfg.eval_steps or self.global_step % cfg.eval_steps != 0:
                self._validate(epoch, callback)

            if self._stop:
                break

        # final validation + best-weight restore ---------------------------------------
        final_metrics = self.evaluate("valid", verbose=False)
        if self._best_state is not None and self.best_valid_loss < float("inf"):
            try:
                self.model.load_state_dict(self._best_state)
            except Exception:  # pragma: no cover - defensive
                pass
        final_metrics_after_restore = self.evaluate("valid", verbose=False)

        result = DPOTrainResult(
            output_dir=cfg.output_dir,
            model_name=cfg.model_name,
            n_train_pairs=len(self.train_dataset) if self.train_dataset else 0,
            n_valid_pairs=len(self.valid_dataset) if self.valid_dataset else 0,
            steps=self.global_step,
            epochs_run=float(self.history[-1]["epoch"]) if self.history else 0.0,
            train_examples_seen=self.examples_seen,
            best_valid_loss=self.best_valid_loss if self.best_valid_loss < float("inf") else final_metrics["loss"],
            best_step=self.best_step,
            final_train_loss=float(self.history[-1]["loss"]) if self.history else float("nan"),
            final_valid_loss=final_metrics_after_restore["loss"],
            final_valid_accuracy=final_metrics_after_restore["accuracy"],
            history=list(self.history),
            config=cfg.to_dict(),
            meta={
                "beta": cfg.beta,
                "optimizer": cfg.optimizer,
                "patience": cfg.patience,
                "validation_metric": cfg.validation_metric,
                "completed": not self._stop,
                "final_metrics": _json_safe(final_metrics_after_restore),
                "elapsed_sec": _now() - start_time,
            },
        )
        return result

    def _validate(self, epoch: int, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        metrics = self.evaluate("valid", verbose=False)
        record = {
            "step": self.global_step,
            "epoch": epoch + 1,
            "loss/valid": metrics["loss"],
            "accuracy/valid": metrics["accuracy"],
            "reward_margin/valid": metrics["reward_margin"],
            "examples_seen": self.examples_seen,
            "split": "valid",
        }
        self.history.append(record)
        if callback is not None:
            callback(record)
        if self.config.verbose:
            print(
                f"[dpo] step {self.global_step} epoch {epoch + 1} "
                f"loss/valid={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                f"examples={self.examples_seen}"
            )

        improved = metrics["loss"] < self.best_valid_loss - 1e-6
        if improved:
            self.best_valid_loss = metrics["loss"]
            self.best_step = self.global_step
            self.evals_without_improvement = 0
            self.save(self.config.output_dir)
            try:
                import copy

                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            except Exception:  # pragma: no cover - defensive
                self._best_state = None
        else:
            self.evals_without_improvement += 1
            if self.config.early_stopping and self.evals_without_improvement >= self.config.patience:
                if self.config.verbose:
                    print(
                        f"[dpo] early stopping: no improvement in "
                        f"{self.config.patience} validations (best={self.best_valid_loss:.4f})"
                    )
                self._stop = True

    # -- persistence -----------------------------------------------------------------
    def save(self, output_dir: Optional[str] = None, save_state: bool = True) -> str:
        """Save the policy model + tokenizer (this is ``GPT2_DPO``)."""
        out = output_dir or self.config.output_dir
        os.makedirs(out, exist_ok=True)
        self.model.save_pretrained(out)
        try:
            self.tokenizer.save_pretrained(out)
        except Exception:  # pragma: no cover - optional
            pass
        if save_state:
            state = {
                "global_step": self.global_step,
                "examples_seen": self.examples_seen,
                "best_valid_loss": None
                if math.isinf(self.best_valid_loss)
                else self.best_valid_loss,
                "best_step": self.best_step,
                "evals_without_improvement": self.evals_without_improvement,
                "config": self.config.to_dict(),
                "history": _json_safe(self.history),
            }
            with open(os.path.join(out, TRAINER_STATE_FILENAME), "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        return out

    @classmethod
    def load(cls, output_dir: str, model=None, ref_model=None, tokenizer=None, config=None):
        """Reload a trainer from a saved directory (policy weights + tokenizer)."""
        from .model_utils import load_model  # lazy

        cfg_path = os.path.join(output_dir, TRAINER_STATE_FILENAME)
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            if config is None:
                config = DPOConfig.from_dict(state.get("config", {}))
            if isinstance(config, DPOConfig):
                config.output_dir = output_dir
        if model is None or tokenizer is None:
            model, tokenizer = load_model(output_dir)
        trainer = cls(model, ref_model, tokenizer, config=config)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            trainer.global_step = int(state.get("global_step", 0))
            trainer.examples_seen = int(state.get("examples_seen", 0))
            best = state.get("best_valid_loss")
            trainer.best_valid_loss = float("inf") if best is None else float(best)
            trainer.best_step = int(state.get("best_step", -1))
            trainer.history = list(state.get("history", []))
        return trainer


# --------------------------------------------------------------------------------------
# Orchestration helpers
# --------------------------------------------------------------------------------------


def _progress(iterable: Iterable, desc: str):
    """tqdm wrapper with a silent fallback."""
    try:  # pragma: no cover - optional dependency
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except Exception:  # pragma: no cover
        return iterable


def _now() -> float:
    import time

    return time.time()


def build_datasets(
    pairs: Any,
    tokenizer,
    valid_pairs: Any = None,
    valid_ratio: float = 0.1,
    seed: int = 0,
    max_length: int = 128,
) -> Tuple[DPODataset, DPODataset]:
    """Build ``(train, valid)`` :class:`DPODataset` objects.

    If only a combined container is given, a deterministic 90:10 split is
    produced via :mod:`data.pairwise` (matching Section 4.2).
    """
    if valid_pairs is not None:
        train_ds = DPODataset(pairs, tokenizer, max_length)
        valid_ds = DPODataset(valid_pairs, tokenizer, max_length)
        return train_ds, valid_ds

    split = None
    if isinstance(pairs, dict) and "train" in pairs and "valid" in pairs:
        train_ds = DPODataset(pairs["train"], tokenizer, max_length)
        valid_ds = DPODataset(pairs["valid"], tokenizer, max_length)
        return train_ds, valid_ds

    try:  # data.pairwise container (has .train/.valid)
        from data import pairwise as pw

        if hasattr(pairs, "train") and hasattr(pairs, "valid"):
            split = pairs
        else:
            pairs_list = _as_pair_dicts(pairs)
            split = pw.build_dataset(pairs_list, valid_ratio=valid_ratio, seed=seed)
        return (
            DPODataset(split.train.to_hf(), tokenizer, max_length),
            DPODataset(split.valid.to_hf(), tokenizer, max_length),
        )
    except Exception:  # pragma: no cover - fallback split
        rows = _as_pair_dicts(pairs)
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
        n_valid = int(len(rows) * valid_ratio)
        return (
            DPODataset(rows[n_valid:], tokenizer, max_length),
            DPODataset(rows[:n_valid], tokenizer, max_length),
        )


def load_pairs_artifact(path: Optional[str] = None):
    """Load pairs from a JSONL artifact (or shards) via :mod:`data.pairwise`."""
    from data import pairwise as pw  # lazy

    path = path or pw.DEFAULT_PAIRS_PATH
    if os.path.exists(path):
        return pw.load_pairwise_dataset(path)
    if os.path.isdir(pw.DEFAULT_SHARD_DIR):
        pairs = pw.load_shards(pw.DEFAULT_SHARD_DIR)
        if pairs:
            return pw.build_dataset(pairs)
    return None


def train_dpo(
    pairs: Any = None,
    pairs_path: Optional[str] = None,
    model=None,
    tokenizer=None,
    ref_model=None,
    config: Any = None,
    model_name: str = GPT2_MEDIUM,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    valid_ratio: float = 0.1,
    seed: int = 0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple["DPOTrainer", DPOTrainResult]:
    """End-to-end DPO training on GPT2-medium (Section 4.2).

    Loads model/tokenizer/pairs when not supplied, splits 90:10, trains with the
    Table 8 hyperparameters and saves the resulting ``GPT2_DPO`` to
    ``output_dir``.
    """
    from .model_utils import load_model, set_seed  # lazy

    cfg = config if isinstance(config, DPOConfig) else DPOConfig.from_dict(config or {})
    cfg.model_name = model_name or cfg.model_name
    cfg.output_dir = output_dir or cfg.output_dir
    cfg.device = device or cfg.device
    cfg.seed = seed if seed is not None else cfg.seed
    set_seed(cfg.seed)

    if model is None or tokenizer is None:
        model, tokenizer = load_model(cfg.model_name, device=cfg.device)

    if ref_model is None:
        import copy

        ref_model = copy.deepcopy(model)
        for p in ref_model.parameters():
            p.requires_grad_(False)

    if pairs is None:
        pairs = load_pairs_artifact(pairs_path)
        if pairs is None:
            raise FileNotFoundError(
                "No preference pairs found. Run scripts/generate_pairs.py first "
                f"(expected {pairs_path or DEFAULT_PAIRS_PATH})."
            )

    train_ds, valid_ds = build_datasets(
        pairs,
        tokenizer,
        valid_ratio=valid_ratio,
        seed=cfg.seed,
        max_length=cfg.max_length,
    )
    if cfg.verbose:
        print(
            f"[dpo] train pairs={len(train_ds)} valid pairs={len(valid_ds)} "
            f"beta={cfg.beta} lr={cfg.learning_rate} batch={cfg.batch_size} "
            f"optimizer={cfg.optimizer}"
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=cfg,
        train_pairs=train_ds.pairs,
        valid_pairs=valid_ds.pairs,
    )
    result = trainer.train()

    if verbose:
        print(f"[dpo] saved GPT2_DPO to {cfg.output_dir}")
        print(f"[dpo] summary: {json.dumps(_json_safe(result.summary()), indent=2)}")
    with open(
        os.path.join(cfg.output_dir, "dpo_train_result.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(result.to_dict(), fh, indent=2)
    return trainer, result


def save_result(path: str, result: DPOTrainResult) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    return path


def load_result(path: str) -> DPOTrainResult:
    with open(path, "r", encoding="utf-8") as fh:
        return DPOTrainResult.from_dict(json.load(fh))


def load_dpo_model(path: str = DEFAULT_OUTPUT_DIR, device: Optional[str] = None):
    """Load a trained ``GPT2_DPO`` checkpoint."""
    from .model_utils import load_model  # lazy

    return load_model(path, device=device)


def pair_logratios(model, ref_model, tokenizer, pairs, max_length: int = 128, device=None):
    """Implicit rewards / log-ratios for analysis (``beta log(pi/pi_ref)``).

    Returns ``{'chosen': ndarray, 'rejected': ndarray, 'margin': ndarray}``.
    """
    import torch  # lazy

    from .model_utils import resolve_device

    dev = resolve_device(device)
    ds = DPODataset(pairs, tokenizer, max_length)
    chosen, rejected = [], []
    model.eval()
    ref_model.eval()
    with torch.no_grad():
        for batch in make_loader(ds, 4, shuffle=False):
            for prefix, label_key, store in (
                ("chosen", "chosen_labels", chosen),
                ("rejected", "rejected_labels", rejected),
            ):
                ids = batch[f"{prefix}_input_ids"].to(dev)
                mask = batch[f"{prefix}_attention_mask"].to(dev)
                labels = batch[label_key].to(dev)
                pol = sequence_logprobs(model, ids, mask, labels)
                ref = sequence_logprobs(ref_model, ids, mask, labels)
                store.append((pol - ref).detach().cpu().numpy())
    import numpy as np  # lazy

    c = np.concatenate(chosen) if chosen else np.zeros(0)
    r = np.concatenate(rejected) if rejected else np.zeros(0)
    return {"chosen": c, "rejected": r, "margin": c - r}


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - manual smoke test
    """Sanity checks for the DPO loss (no network / GPU required)."""
    import torch

    # Eq. 1 sanity: identical policy/reference -> logits 0 -> loss = log 2.
    z = torch.zeros(4)
    loss, cr, rr = dpo_loss(z, z, z, z, beta=DPO_BETA)
    assert abs(float(loss) - math.log(2.0)) < 1e-6, loss
    # Perfect policy: huge positive margin -> loss ~ 0.
    good = torch.full((4,), 100.0)
    loss_good, _, _ = dpo_loss(good, z, z, z, beta=DPO_BETA)
    assert float(loss_good) < 1e-6, loss_good
    # Reversed policy -> large loss (~ beta * margin).
    loss_bad, _, _ = dpo_loss(z, good, z, z, beta=DPO_BETA)
    assert float(loss_bad) > 5.0, loss_bad
    print("[dpo_trainer] loss sanity checks passed")
    print(f"  log(2)={math.log(2.0):.6f}  identical-loss={float(loss):.6f}")
    print(f"  perfect-policy-loss={float(loss_good):.3e}  reversed-loss={float(loss_bad):.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
