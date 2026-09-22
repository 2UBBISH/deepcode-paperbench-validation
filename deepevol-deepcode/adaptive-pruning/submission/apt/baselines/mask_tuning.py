"""Mask Tuning (Kwon et al., 2022) baseline -- the paper's "LoRA+Prune" row.

Paper reference (Section 5.2, *Baselines*)::

    LoRA+Prune: a post-training pruning method over on LoRA-tuned LMs. We use
    Mask Tuning (Kwon et al., 2022), a state-of-the-art post-training structured
    pruning method based on fisher information. Due to that post-training pruning
    performs poorly on high-sparsity settings, we retrain the pruned LM after
    pruning to recover its performance.

The official implementation lives in
`WoosukKwon/retraining-free-pruning <https://github.com/WoosukKwon/retraining-free-pruning>`_.
This module

1. locates / (optionally) clones that repository and reports availability
   (:func:`locate_external_repo`, :func:`external_repo_available`,
   :func:`load_external_repo`, :func:`clone_external_repo`), and
2. provides a faithful in-repo adaptation of Mask Tuning's *Fisher-information
   based structured pruning* so the Table 2 ``LoRA+Prune`` row can be produced
   even without a network clone:

   * stage 1 -- LoRA-style tuning of the LM (adapters on Query/Value, plus the
     FFN projections for models where APT also tunes them),
   * stage 2 -- mask pruning: Fisher information of the *mask variables* is
     accumulated over the training set and groups of parameters (MHA heads, FFN
     neurons, hidden dimensions) are ranked by ``salience / parameter count``
     (density) under the sparsity constraint,
   * stage 3 -- retraining (*recovery*) of the pruned, physically smaller LM.

Two scoring modes are supported: ``"fisher"`` (the paper's squared-gradient
mask salience, aggregated per block) and ``"magnitude"`` (weight-magnitude
fallback used when gradients are unavailable).

Public entry points
-------------------
``MaskTuningConfig`` / ``MaskTuningMethod`` / ``train_mask_tuning`` /
``prune_with_mask_tuning`` / ``evaluate_mask_tuning`` / ``external_repo_available``
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is optional so the module imports in bare envs
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


__all__ = [
    "TTA_FRACTION",
    "MASK_TUNING_REPO_URL",
    "MASK_TUNING_REPO_DIRNAME",
    "DEFAULT_FISHER_NUM_BATCHES",
    "TABLE6_MASK_TUNING_DEFAULTS",
    "SCORING_MODES",
    "MaskTuningConfig",
    "MaskTuningMethod",
    "FisherScoreSet",
    "MaskTuningPruneResult",
    "external_repo_available",
    "locate_external_repo",
    "load_external_repo",
    "clone_external_repo",
    "compute_fisher_scores",
    "block_scores_from_fisher",
    "select_blocks_from_fisher",
    "apply_binary_masks",
    "prune_with_mask_tuning",
    "train_mask_tuning",
    "evaluate_mask_tuning",
    "build_model_and_tokenizer",
    "mask_tuning_trainer",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TTA_FRACTION = 0.97
"""Paper protocol: time-to-accuracy is measured to 97% of fully fine-tuned perf."""

MASK_TUNING_REPO_URL = "https://github.com/WoosukKwon/retraining-free-pruning"
MASK_TUNING_REPO_DIRNAME = "retraining-free-pruning"

DEFAULT_FISHER_NUM_BATCHES = 32
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_TARGET_SPARSITY = 0.60
DEFAULT_LORA_RANK = 8
DEFAULT_SCALING = 2.0

SCORING_MODES = ("fisher", "magnitude", "auto")

GLUE_TASKS = ("mnli", "sst2", "qnli", "qqp", "mrpc", "cola", "rte", "stsb")
GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad", "squad_v2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

GLUE_NUM_LABELS = {
    "mnli": 3,
    "sst2": 2,
    "qnli": 2,
    "qqp": 2,
    "mrpc": 2,
    "cola": 2,
    "rte": 2,
    "stsb": 1,
}
GLUE_REGRESSION_TASKS = ("stsb", "sts-b")

#: Table 6 columns reused for the pruning baselines.  ``epochs`` is the *total*
#: budget: ``distill_epochs`` are spent on the initial LoRA tuning and the
#: remainder on post-pruning recovery retraining (paper: "we retrain the pruned
#: LM after pruning to recover its performance").
TABLE6_MASK_TUNING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "glue-big": {"learning_rate": 2.0e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "glue-small": {"learning_rate": 2.0e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "squad": {"learning_rate": 2.0e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "cnndm": {"learning_rate": 1.0e-4, "batch_size": 16, "epochs": 16, "distill_epochs": 6},
}

MODEL_INPUT_KEYS: Tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
    "inputs_embeds",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
    "start_positions",
    "end_positions",
    "p_mask",
    "cls_index",
)

# block-type constants (mirror apt.adapters / apt.block_selection)
HEAD = 0
NEURON = 1
DIMENSION = 2
KIND_NAMES = {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}

_TASK_ALIASES = {
    "sst-2": "sst2",
    "sst_2": "sst2",
    "sst": "sst2",
    "mnli-mm": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatched": "mnli",
    "sts-b": "stsb",
    "stsb": "stsb",
    "cola": "cola",
    "qqp": "qqp",
    "qnli": "qnli",
    "rte": "rte",
    "mrpc": "mrpc",
    "squad2": "squad_v2",
    "squad-v2": "squad_v2",
    "squad_v2": "squad_v2",
    "squadv2": "squad_v2",
    "cnn_dailymail": "cnndm",
    "cnn-dailymail": "cnndm",
    "cnndm": "cnndm",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _torch():
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("apt.baselines.mask_tuning requires PyTorch for the pruning pipeline.")
    return torch


def canonical_task(task: Optional[str]) -> str:
    """Normalise a task name (``SST-2`` -> ``sst2``)."""
    if task is None:
        return "sst2"
    key = str(task).strip().lower().replace(" ", "")
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    return _TASK_ALIASES.get(key.replace("-", "_"), key.replace("-", "_"))


def is_glue_task(task: Optional[str]) -> bool:
    return canonical_task(task) in GLUE_TASKS


def is_squad_task(task: Optional[str]) -> bool:
    return canonical_task(task) in SQUAD_TASKS


def is_seq2seq_task(task: Optional[str]) -> bool:
    return canonical_task(task) in SEQ2SEQ_TASKS


def num_labels_for_task(task: Optional[str]) -> int:
    return int(GLUE_NUM_LABELS.get(canonical_task(task), 2))


def problem_type_for_task(task: Optional[str]) -> Optional[str]:
    return "regression" if canonical_task(task) in GLUE_REGRESSION_TASKS else None


def table6_group_for(model_type: Optional[str], task: Optional[str]) -> str:
    task = canonical_task(task)
    if task in SEQ2SEQ_TASKS:
        return "cnndm"
    if task in SQUAD_TASKS:
        return "squad"
    return "glue-big" if task in GLUE_BIG_TASKS else "glue-small"


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    try:  # pragma: no cover - numpy optional
        import numpy as _np

        _np.random.seed(int(seed))
    except Exception:
        pass
    if _HAS_TORCH:  # pragma: no cover
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: Optional[Any] = None):
    if not _HAS_TORCH:
        return None
    if device is not None:
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(device)
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: Any, device: Any) -> Any:
    if not _HAS_TORCH or device is None:
        return batch
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def model_inputs(batch: Any, keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, Any]:
    """Keep only entries that are valid HuggingFace ``forward`` arguments."""
    if not isinstance(batch, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in keys:
        if key in batch and batch[key] is not None:
            out[key] = batch[key]
    if not out:  # fall back to every tensor-valued entry
        for key, value in batch.items():
            if _HAS_TORCH and torch.is_tensor(value):
                out[key] = value
    return out


def _shape_get(shape: Any, name: str, default: Any = None) -> Any:
    if shape is None:
        return default
    if isinstance(shape, dict):
        value = shape.get(name, default)
    else:
        value = getattr(shape, name, default)
    return default if value is None else value


def _weight_of(module: Any):
    weight = getattr(module, "base_weight", None)
    if weight is None:
        weight = getattr(module, "weight", None)
    return weight


def _row_unit_count(module: Any, fallback: int) -> int:
    try:
        return int(_weight_of(module).shape[0])
    except Exception:
        return int(fallback)


def _col_unit_count(module: Any, fallback: int) -> int:
    try:
        return int(_weight_of(module).shape[1])
    except Exception:
        return int(fallback)


def _to_list(tensor: Any) -> List[float]:
    if tensor is None:
        return []
    try:
        return [float(v) for v in tensor.detach().to(torch.float32).reshape(-1).cpu().tolist()]
    except Exception:
        try:
            return [float(v) for v in list(tensor)]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# external repository handling
# ---------------------------------------------------------------------------


def _looks_like_repo(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    markers = (
        "README.md",
        "mask_tuning",
        "mask_tuning.py",
        "prune.py",
        "main.py",
        "utils.py",
        "lib",
        "single_layer",
        "requirements.txt",
    )
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def locate_external_repo(
    path: Optional[str] = None, *, extra_candidates: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Return the checkout directory of the Mask Tuning repository, if present."""
    candidates: List[str] = []
    if path:
        candidates.append(path)
    for env_key in ("MASK_TUNING_REPO", "RETRAINING_FREE_PRUNING", "MASKTUNING_ROOT"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(env_value)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates.extend(
        [
            os.path.join(root, "external", MASK_TUNING_REPO_DIRNAME),
            os.path.join(root, "third_party", MASK_TUNING_REPO_DIRNAME),
            os.path.join(root, MASK_TUNING_REPO_DIRNAME),
            os.path.join(root, "external", "mask_tuning"),
            os.path.join(os.path.expanduser("~"), MASK_TUNING_REPO_DIRNAME),
            os.path.join("/opt", MASK_TUNING_REPO_DIRNAME),
        ]
    )
    if extra_candidates:
        candidates.extend(list(extra_candidates))
    for candidate in candidates:
        if _looks_like_repo(candidate):
            return os.path.abspath(candidate)
    return None


_EXTERNAL_STATE: Dict[str, Any] = {"path": None, "module": None, "checked": False}

_EXTERNAL_MODULE_CANDIDATES = (
    "mask_tuning",
    "prune",
    "pruning",
    "retraining_free_pruning",
    "fisher",
    "lib.mask_tuning",
)


def load_external_repo(
    path: Optional[str] = None,
    *,
    force: bool = False,
    module_names: Optional[Sequence[str]] = None,
) -> Optional[ModuleType]:
    """Import a module from the Mask Tuning repository (best effort).

    Returns ``None`` when the repository is absent or none of the known entry
    points can be imported.  The result is memoised; pass ``force=True`` to
    re-scan.
    """
    if _EXTERNAL_STATE["checked"] and not force:
        return _EXTERNAL_STATE["module"]
    _EXTERNAL_STATE["checked"] = True
    repo = locate_external_repo(path)
    _EXTERNAL_STATE["path"] = repo
    _EXTERNAL_STATE["module"] = None
    if repo is None:
        return None
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for name in tuple(module_names) if module_names else _EXTERNAL_MODULE_CANDIDATES:
        try:  # pragma: no cover - depends on the external checkout
            module = __import__(name, fromlist=["*"])
        except Exception:
            continue
        _EXTERNAL_STATE["module"] = module
        return module
    return None


def external_repo_available(path: Optional[str] = None) -> bool:
    """``True`` when the Mask Tuning repository can be located or imported."""
    if load_external_repo(path) is not None:
        return True
    return locate_external_repo(path) is not None


def clone_external_repo(
    dest: Optional[str] = None,
    *,
    url: str = MASK_TUNING_REPO_URL,
    depth: int = 1,
    verbose: bool = True,
) -> Optional[str]:
    """Clone the Mask Tuning repository (requires network access).

    Returns the checkout directory, or ``None`` if cloning failed.
    """
    if dest is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        dest = os.path.join(root, "external", MASK_TUNING_REPO_DIRNAME)
    if _looks_like_repo(dest):
        return os.path.abspath(dest)
    if shutil.which("git") is None:
        if verbose:
            warnings.warn("git executable not found; cannot clone Mask Tuning.")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    cmd = ["git", "clone", "--depth", str(int(depth)), url, dest]
    try:  # pragma: no cover - network dependent
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover
        if verbose:
            warnings.warn(f"failed to clone {url}: {exc}")
        return None
    if verbose:
        print(f"[mask_tuning] cloned {url} -> {dest}")
    return os.path.abspath(dest) if _looks_like_repo(dest) else None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class MaskTuningConfig:
    """Hyper-parameters for the ``LoRA+Prune`` (Mask Tuning) baseline.

    Defaults follow Table 6 of the APT paper.
    """

    model_name_or_path: str = "roberta-base"
    model_type: Optional[str] = None
    task: str = "sst2"
    table6_group: Optional[str] = None

    # Table 6
    learning_rate: float = 2.0e-4
    batch_size: int = 32
    epochs: int = 40
    distill_epochs: int = 20

    # pruning
    target_sparsity: float = DEFAULT_TARGET_SPARSITY
    target_density: Optional[float] = None
    prune_heads: bool = True
    prune_neurons: bool = True
    prune_dims: bool = True

    # scoring
    scoring: str = "fisher"
    fisher_num_batches: int = DEFAULT_FISHER_NUM_BATCHES
    fisher_pool: str = "token"
    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    use_external_repo: bool = True
    external_repo_path: Optional[str] = None

    # LoRA-style tuning of the (unpruned) LM
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: Optional[float] = None
    scaling: float = DEFAULT_SCALING
    lora_dropout: float = 0.0
    lora_target_modules: Optional[Sequence[str]] = None
    tune_ffn_adapters: bool = True

    # optimisation
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.06
    lr_kind: str = "linear"
    max_grad_norm: float = 1.0
    seed: Optional[int] = 42

    # data
    max_seq_length: int = 128
    max_target_length: int = 128
    doc_stride: int = 128
    max_query_length: int = 64
    n_best_size: int = 20
    max_answer_length: int = 30
    null_score_diff_threshold: float = 0.0
    dynamic_padding: bool = False
    num_workers: int = 0

    # runtime / reporting
    device: Optional[str] = "cuda"
    output_dir: str = "outputs/mask_tuning"
    logging_steps: int = 50
    eval_steps: int = 0
    save_steps: int = 0
    inference_batch_size: int = 128
    sequence_length: int = 128
    fp16: bool = False
    bf16: bool = False
    measure_efficiency: bool = True
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "MaskTuningConfig":
        payload: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        if data:
            for key, value in dict(data).items():
                if key in cls.__dataclass_fields__:
                    payload[key] = value
                else:
                    extras[key] = value
        if extras:
            payload.setdefault("extra", {}).update(extras)
        payload.update({k: v for k, v in overrides.items() if v is not None})
        config = cls(**payload)
        if config.table6_group is None:
            config.table6_group = table6_group_for(config.model_type, config.task)
        return config

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "MaskTuningConfig":
        import yaml  # local import: optional dependency

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data, **overrides)

    # -- helpers ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
        return path

    @property
    def is_squad(self) -> bool:
        return is_squad_task(self.task)

    @property
    def is_seq2seq(self) -> bool:
        return is_seq2seq_task(self.task)

    @property
    def num_epochs(self) -> int:
        return int(self.epochs)

    @property
    def lora_epochs(self) -> int:
        """Epochs of the initial LoRA tuning (= ``distill_epochs``)."""
        epochs = int(self.distill_epochs or 0)
        return max(1, min(epochs, self.num_epochs))

    @property
    def recovery_epochs(self) -> int:
        """Post-pruning retraining epochs (paper: retrain after pruning)."""
        return max(1, self.num_epochs - self.lora_epochs)

    @property
    def resolved_density(self) -> float:
        if self.target_density is not None:
            return float(max(0.0, min(1.0, self.target_density)))
        return float(max(0.0, min(1.0, 1.0 - self.target_sparsity)))

    @property
    def resolved_sparsity(self) -> float:
        return float(max(0.0, min(1.0, 1.0 - self.resolved_density)))

    @property
    def resolved_scaling(self) -> float:
        if self.lora_alpha is not None and self.lora_rank:
            return float(self.lora_alpha) / float(self.lora_rank)
        return float(self.scaling)

    @property
    def resolved_target_modules(self) -> Optional[Sequence[str]]:
        return list(self.lora_target_modules) if self.lora_target_modules else None


def apply_table6_defaults(
    config: MaskTuningConfig, table: Optional[Dict[str, Dict[str, Any]]] = None
) -> MaskTuningConfig:
    """Fill Table 6 values for the config's ``table6_group``."""
    table = table or TABLE6_MASK_TUNING_DEFAULTS
    group = config.table6_group or table6_group_for(config.model_type, config.task)
    config.table6_group = group
    defaults = table.get(group)
    if not defaults:
        return config
    if config.learning_rate == MaskTuningConfig.learning_rate:
        config.learning_rate = float(defaults.get("learning_rate", config.learning_rate))
    if config.batch_size == MaskTuningConfig.batch_size:
        config.batch_size = int(defaults.get("batch_size", config.batch_size))
    if config.epochs == MaskTuningConfig.epochs:
        config.epochs = int(defaults.get("epochs", config.epochs))
    if config.distill_epochs == MaskTuningConfig.distill_epochs:
        config.distill_epochs = int(defaults.get("distill_epochs", config.distill_epochs))
    if group == "cnndm":
        config.max_seq_length = max(int(config.max_seq_length), 512)
        config.max_target_length = max(int(config.max_target_length), 128)
    return config


# ---------------------------------------------------------------------------
# model / tokenizer / data construction
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(config: MaskTuningConfig):
    """Instantiate the HuggingFace (model, tokenizer) pair for ``config.task``."""
    from transformers import AutoTokenizer

    name = config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    task = canonical_task(config.task)
    if is_squad_task(task):
        from transformers import AutoModelForQuestionAnswering as ModelCls

        model = ModelCls.from_pretrained(name)
    elif is_seq2seq_task(task):
        from transformers import AutoModelForSeq2SeqLM as ModelCls

        model = ModelCls.from_pretrained(name)
    else:
        from transformers import AutoModelForSequenceClassification as ModelCls

        kwargs: Dict[str, Any] = {"num_labels": num_labels_for_task(task)}
        problem_type = problem_type_for_task(task)
        if problem_type:
            kwargs["problem_type"] = problem_type
        model = ModelCls.from_pretrained(name, **kwargs)
    return model, tokenizer


def build_dataloaders_for_task(
    config: MaskTuningConfig, tokenizer: Any, splits: Sequence[str] = ("train", "validation")
):
    """Build task dataloaders through :mod:`apt.data` (Table 6 batch sizes)."""
    from apt.data import make_dataloaders

    kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "model_type": "t5" if is_seq2seq_task(config.task) else "encoder",
        "splits": tuple(splits),
        "num_workers": config.num_workers,
        "seed": config.seed if config.seed is not None else 42,
    }
    if is_squad_task(config.task):
        kwargs.update(
            {
                "max_seq_length": config.max_seq_length,
                "doc_stride": config.doc_stride,
                "max_query_length": config.max_query_length,
            }
        )
    elif is_seq2seq_task(config.task):
        kwargs.update(
            {
                "max_source_length": config.max_seq_length,
                "max_target_length": config.max_target_length,
            }
        )
    else:
        kwargs.update({"max_seq_length": config.max_seq_length})
    return make_dataloaders(config.task, tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# Fisher scoring
# ---------------------------------------------------------------------------


@dataclass
class _ModuleRecord:
    name: str
    module: Any
    kind: int
    layer: int
    site: str
    group: int
    d_in: int
    d_out: int


@dataclass
class FisherScoreSet:
    """Per-module Fisher information accumulated over the calibration set."""

    rows: Dict[str, Any] = field(default_factory=dict)
    cols: Dict[str, Any] = field(default_factory=dict)
    records: List[_ModuleRecord] = field(default_factory=list)
    d_model: int = 0
    n_batches: int = 0
    scoring: str = "fisher"

    def row(self, name: str, index: int) -> float:
        tensor = self.rows.get(name)
        if tensor is None:
            return 0.0
        try:
            return float(tensor[int(index)])
        except Exception:
            return 0.0

    def col(self, name: str, index: int) -> float:
        tensor = self.cols.get(name)
        if tensor is None:
            return 0.0
        try:
            return float(tensor[int(index)])
        except Exception:
            return 0.0

    def as_dict(self, digits: int = 6) -> Dict[str, Any]:
        return {
            "n_batches": self.n_batches,
            "scoring": self.scoring,
            "modules": [
                {
                    "name": rec.name,
                    "kind": KIND_NAMES.get(rec.kind, str(rec.kind)),
                    "layer": rec.layer,
                    "site": rec.site,
                    "rows": [round(v, digits) for v in _to_list(self.rows.get(rec.name, []))],
                    "cols": [round(v, digits) for v in _to_list(self.cols.get(rec.name, []))],
                }
                for rec in self.records
            ],
        }


def module_records(model: Any) -> List[_ModuleRecord]:
    """Enumerate the wrapped projections of ``model`` with their block metadata."""
    records: List[_ModuleRecord] = []
    try:
        from apt.adapters import iter_masked_linears
    except Exception:
        return records
    try:
        from apt.salience import module_site
    except Exception:  # pragma: no cover - salience is optional
        def module_site(name, kind=HEAD):  # type: ignore[misc]
            lowered = str(name).lower()
            if kind == HEAD:
                return "value" if ("value" in lowered or lowered.endswith("v")) else "query"
            if kind == NEURON:
                return "ffn_in"
            return "hidden"

    for name, module in iter_masked_linears(model):
        kind = int(getattr(module, "kind", HEAD))
        records.append(
            _ModuleRecord(
                name=str(name),
                module=module,
                kind=kind,
                layer=int(getattr(module, "layer_idx", -1)),
                site=str(module_site(name, kind)),
                group=max(1, int(getattr(module, "out_group_size", 1) or 1)),
                d_out=_row_unit_count(module, 0),
                d_in=_col_unit_count(module, 0),
            )
        )
    return records


def _pooled_fisher(x: Any, g: Any, *, pool: str = "token", chunk: int = 8) -> Tuple[Any, Any]:
    """Fisher information of the weight/mask variables from cached ``(x, dL/dz)``.

    ``rows`` scores output units (heads / FFN neurons) and ``cols`` scores input
    units (hidden dimensions).  As in the paper's salience convention the
    products are reduced over the batch *and* sequence before the squared
    aggregation, which keeps memory bounded (AdaPrune/Mask Tuning style).
    """
    torch = _torch()
    x = x.detach().to(torch.float32)
    g = g.detach().to(torch.float32)
    if x.dim() > 2 or g.dim() > 2:
        if pool == "sample" and x.dim() >= 3 and g.shape == x.shape:
            rows = None
            cols = None
            n_samples = 0
            for start in range(0, x.shape[0], max(1, chunk)):
                xb = x[start : start + chunk]
                gb = g[start : start + chunk]
                flat_x = xb.reshape(xb.shape[0], -1, xb.shape[-1])
                flat_g = gb.reshape(gb.shape[0], -1, gb.shape[-1])
                if flat_x.shape[1] != flat_g.shape[1]:
                    break
                prod = flat_x.transpose(1, 2) @ flat_g
                prod = prod * prod
                rows_b = prod.sum(dim=1)
                cols_b = prod.sum(dim=2)
                rows = rows_b if rows is None else rows + rows_b
                cols = cols_b if cols is None else cols + cols_b
                n_samples += int(prod.shape[0])
            if rows is not None and n_samples:
                return rows / float(n_samples), cols / float(n_samples)
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
    if x.shape[0] != g.shape[0]:
        keep = min(int(x.shape[0]), int(g.shape[0]))
        x = x[:keep]
        g = g[:keep]
    prod = x.transpose(0, 1) @ g
    prod = prod * prod
    return prod.sum(dim=0), prod.sum(dim=1)


def _module_magnitude_scores(record: _ModuleRecord) -> Tuple[Any, Any]:
    """Weight-magnitude fallback used when gradients are unavailable."""
    torch = _torch()
    weight = _weight_of(record.module)
    if weight is None:
        return None, None
    w = weight.detach().to(torch.float32)
    return (w * w).sum(dim=1), (w * w).sum(dim=0)


def _task_loss(outputs: Any, inputs: Dict[str, Any]):
    """Return the task loss from HF outputs (falls back to a manual loss)."""
    torch = _torch()
    loss = getattr(outputs, "loss", None)
    if loss is not None and torch.is_tensor(loss):
        return loss
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if logits is None or not torch.is_tensor(logits):
        return None
    labels = inputs.get("labels")
    if labels is None and "start_positions" in inputs and "end_positions" in inputs:
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        if logits.dim() == 4 and logits.shape[-1] == 2:  # QA models: (2, b, s)
            start_logits, end_logits = logits[..., 0], logits[..., 1]
        elif logits.dim() == 3 and logits.shape[-1] == 2:
            start_logits, end_logits = logits[..., 0], logits[..., 1]
        else:
            start_logits, end_logits = logits
        return torch.nn.functional.cross_entropy(start_logits, start_positions) + torch.nn.functional.cross_entropy(
            end_logits, end_positions
        )
    if labels is None:
        return None
    labels = labels.to(logits.device)
    if logits.dim() == 3 and labels.dim() == 2:
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
    if logits.dim() == labels.dim() + 1:
        return torch.nn.functional.cross_entropy(logits, labels)
    return torch.nn.functional.mse_loss(logits.reshape(labels.shape).float(), labels.float())


def compute_fisher_scores(
    model: Any,
    dataloader: Any,
    *,
    task: str = "sst2",
    device: Any = None,
    num_batches: Optional[int] = DEFAULT_FISHER_NUM_BATCHES,
    pool: str = "token",
    scoring: str = "auto",
    max_batches: Optional[int] = None,
    verbose: bool = False,
) -> FisherScoreSet:
    """Accumulate Fisher information over ``num_batches`` calibration batches.

    ``scoring="auto"`` performs a real backward pass when gradients flow and
    falls back to weight magnitude otherwise.
    """
    torch = _torch()
    device = resolve_device(device)
    if dataloader is None:
        raise ValueError("compute_fisher_scores requires a calibration dataloader")
    records = module_records(model)
    if not records:
        raise RuntimeError(
            "no wrapped (masked) linear layers found; wrap the model with "
            "apt.model_wrapper.wrap_model before computing Fisher scores"
        )
    score_set = FisherScoreSet(records=records, scoring="fisher")
    for record in records:
        score_set.rows[record.name] = None
        score_set.cols[record.name] = None

    was_training = bool(getattr(model, "training", False))
    model.eval()
    n_used = 0
    limit = int(num_batches) if num_batches else None
    try:
        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= int(max_batches):
                break
            if limit is not None and n_used >= limit:
                break
            inputs = model_inputs(batch)
            if not inputs:
                continue
            inputs = move_to_device(inputs, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = _task_loss(outputs, inputs)
            if loss is None:
                continue
            loss.backward()
            with torch.no_grad():
                for record in records:
                    x = getattr(record.module, "_cached_input", None)
                    g = getattr(record.module, "_cached_output_grad", None)
                    if x is None or g is None:
                        continue
                    try:
                        rows, cols = _pooled_fisher(x, g, pool=pool)
                    except Exception:
                        continue
                    if rows is None or cols is None:
                        continue
                    current_rows = score_set.rows[record.name]
                    current_cols = score_set.cols[record.name]
                    if current_rows is None:
                        score_set.rows[record.name] = rows.detach().cpu()
                        score_set.cols[record.name] = cols.detach().cpu()
                    else:
                        score_set.rows[record.name] = current_rows + rows.detach().cpu()
                        score_set.cols[record.name] = current_cols + cols.detach().cpu()
            n_used += 1
            if verbose and limit and n_used % max(1, limit // 4 or 1) == 0:
                print(f"[mask_tuning] fisher batches: {n_used}/{limit}")
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    missing = [rec.name for rec in records if score_set.rows.get(rec.name) is None]
    if missing and str(scoring) in ("magnitude", "auto"):
        for record in records:
            if score_set.rows.get(record.name) is not None:
                continue
            rows, cols = _module_magnitude_scores(record)
            if rows is None or cols is None:
                continue
            score_set.rows[record.name] = rows.detach().cpu()
            score_set.cols[record.name] = cols.detach().cpu()
            score_set.scoring = "magnitude"
    score_set.n_batches = n_used
    if not any(score_set.rows.get(r.name) is not None for r in records):
        raise RuntimeError("Fisher scoring produced no usable scores")
    score_set.d_model = max([rec.d_in for rec in records] or [0])
    return score_set


def block_scores_from_fisher(
    scores: FisherScoreSet,
    *,
    d_model: int = 0,
    dim_units: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int, int, Optional[str]], float], Dict[int, float]]:
    """Aggregate per-unit Fisher scores into per-block salience.

    Returns ``(block_scores, dim_scores)`` where ``block_scores`` is keyed by
    ``(kind, layer, index, site)`` for head/neuron blocks (plus a site-agnostic
    ``(kind, layer, index, None)`` entry) and ``dim_scores`` maps a hidden
    dimension index to its aggregated score.
    """
    torch = _torch()
    block_scores: Dict[Tuple[int, int, int, Optional[str]], float] = {}
    d_model = int(d_model or scores.d_model or 0)

    def _tensor(record: _ModuleRecord):
        value = scores.rows.get(record.name)
        if value is None:
            return None
        return value.detach().to(torch.float32).reshape(-1)

    for record in scores.records:
        tensor = _tensor(record)
        if tensor is None:
            continue
        if record.kind == HEAD:
            group = max(1, int(record.group))
            n_groups = int(math.ceil(int(tensor.shape[0]) / float(group)))
            for head_idx in range(n_groups):
                lo, hi = head_idx * group, min(int(tensor.shape[0]), (head_idx + 1) * group)
                if hi <= lo:
                    continue
                value = float(tensor[lo:hi].sum())
                block_scores[(HEAD, record.layer, head_idx, record.site)] = (
                    block_scores.get((HEAD, record.layer, head_idx, record.site), 0.0) + value
                )
                block_scores[(HEAD, record.layer, head_idx, None)] = (
                    block_scores.get((HEAD, record.layer, head_idx, None), 0.0) + value
                )
        elif record.kind == NEURON:
            for index in range(int(tensor.shape[0])):
                value = float(tensor[index])
                block_scores[(NEURON, record.layer, index, record.site)] = (
                    block_scores.get((NEURON, record.layer, index, record.site), 0.0) + value
                )
                block_scores[(NEURON, record.layer, index, None)] = (
                    block_scores.get((NEURON, record.layer, index, None), 0.0) + value
                )

    dim_scores: Dict[int, float] = {}
    for record in scores.records:
        if d_model and record.d_in != d_model:
            continue
        tensor = scores.cols.get(record.name)
        if tensor is None:
            continue
        tensor = tensor.detach().to(torch.float32).reshape(-1)
        for index in range(int(tensor.shape[0])):
            dim_scores[index] = dim_scores.get(index, 0.0) + float(tensor[index])
    if verbose:
        print(f"[mask_tuning] aggregated {len(block_scores)} head/neuron blocks, {len(dim_scores)} dimensions")
    return block_scores, dim_scores


# ---------------------------------------------------------------------------
# block selection (density sorting under the sparsity constraint)
# ---------------------------------------------------------------------------


@dataclass
class _FallbackBlock:
    kind: int
    layer: int
    index: int
    site: Optional[str]
    salience: float
    param_count: float

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, str(self.kind))

    @property
    def density(self) -> float:
        return float(self.salience) / float(self.param_count) if self.param_count else 0.0

    @property
    def name(self) -> str:
        return f"{self.kind_name}.{self.layer}.{self.index}.{self.site}" if self.site else f"{self.kind_name}.{self.layer}.{self.index}"

    @property
    def sort_key(self) -> Tuple[float, int, int, int]:
        return (-self.density, self.kind, self.layer, self.index)


@dataclass
class _FallbackSelection:
    retained: List[_FallbackBlock]
    pruned: List[_FallbackBlock]
    original_param_count: float
    param_count: float

    def sparsity(self) -> float:
        if not self.original_param_count:
            return 0.0
        return 1.0 - float(self.param_count) / float(self.original_param_count)


def _block_costs(d_model: int, n_heads: int, n_ffn: int, n_layers: int, ffn_linear_count: int, attn_linear_count: int):
    """Appendix C parameter-count estimates (Eq. 6 constituents).

    RoBERTa-base (d_m=768, n_h=12, d_h=64, n_f=3072, n_L=12, 2 FFN / 4 attention
    linear layers) gives ``C_head=196608``, ``C_neuron=1536``,
    ``C_dimension=110592``.
    """
    try:
        from apt.block_selection import dimension_param_count, head_param_count, neuron_param_count

        return (
            float(head_param_count(d_model, n_heads, attn_linear_count=attn_linear_count)),
            float(neuron_param_count(d_model, ffn_linear_count=ffn_linear_count)),
            float(
                dimension_param_count(
                    d_model,
                    n_ffn,
                    n_layers,
                    ffn_linear_count=ffn_linear_count,
                    attn_linear_count=attn_linear_count,
                )
            ),
        )
    except Exception:
        head_dim = int(d_model // max(1, n_heads))
        return (
            float(attn_linear_count * d_model * head_dim),
            float(ffn_linear_count * d_model),
            float(n_layers * (attn_linear_count * d_model + ffn_linear_count * n_ffn)),
        )


def _fallback_select(
    block_scores: Dict[Tuple[int, int, int, Optional[str]], float],
    dim_scores: Dict[int, float],
    *,
    d_model: int,
    n_heads: int,
    n_ffn: int,
    n_layers: int,
    ffn_linear_count: int,
    attn_linear_count: int,
    sparsity: float,
    prune_heads: bool = True,
    prune_neurons: bool = True,
    prune_dims: bool = True,
) -> _FallbackSelection:
    """Greedy density-sorted equivalent of the latency-saliency knapsack."""
    head_cost, neuron_cost, dim_cost = _block_costs(
        d_model, n_heads, n_ffn, n_layers, ffn_linear_count, attn_linear_count
    )
    blocks: List[_FallbackBlock] = []
    if prune_heads:
        for layer in range(int(n_layers)):
            for head in range(int(n_heads)):
                salience = block_scores.get((HEAD, layer, head, None))
                if salience is None:
                    salience = sum(v for (k, l, i, _s), v in block_scores.items() if k == HEAD and l == layer and i == head)
                blocks.append(_FallbackBlock(HEAD, layer, head, None, float(salience), head_cost))
    if prune_neurons:
        for layer in range(int(n_layers)):
            for neuron in range(int(n_ffn)):
                salience = block_scores.get((NEURON, layer, neuron, None))
                if salience is None:
                    salience = sum(v for (k, l, i, _s), v in block_scores.items() if k == NEURON and l == layer and i == neuron)
                blocks.append(_FallbackBlock(NEURON, layer, neuron, None, float(salience), neuron_cost))
    if prune_dims:
        for index in range(int(d_model)):
            blocks.append(_FallbackBlock(DIMENSION, -1, index, None, float(dim_scores.get(index, 0.0)), dim_cost))

    total = 0.0
    if prune_heads:
        total += float(n_layers) * float(n_heads) * head_cost
    if prune_neurons:
        total += float(n_layers) * float(n_ffn) * neuron_cost
    if prune_dims:
        total += float(d_model) * dim_cost
    if total <= 0.0:
        total = float(sum(b.param_count for b in blocks))

    blocks.sort(key=lambda b: b.sort_key)
    budget = float(sparsity) * total
    pruned: List[_FallbackBlock] = []
    retained: List[_FallbackBlock] = []
    spent = 0.0
    for block in blocks:
        if spent + block.param_count <= budget + 1e-6:
            pruned.append(block)
            spent += block.param_count
        else:
            retained.append(block)
    return _FallbackSelection(
        retained=retained,
        pruned=pruned,
        original_param_count=total,
        param_count=total - spent,
    )


def select_blocks_from_fisher(
    model: Any,
    scores: FisherScoreSet,
    *,
    sparsity: float = DEFAULT_TARGET_SPARSITY,
    prune_heads: bool = True,
    prune_neurons: bool = True,
    prune_dims: bool = True,
    verbose: bool = False,
    **shape_overrides: Any,
):
    """Rank head/neuron/dimension blocks by density and select the top-i set.

    Uses :class:`apt.block_selection.BlockSelector` (APT's latency-saliency
    knapsack implementation) when available and a self-contained greedy
    equivalent otherwise.
    """
    shape = shape_overrides.pop("shape", None)
    if shape is None:
        try:
            from apt.model_wrapper import apt_shape

            shape = apt_shape(model)
        except Exception:
            shape = None
    d_model = int(_shape_get(shape, "d_model", shape_overrides.pop("d_model", 768)) or 768)
    n_heads = int(_shape_get(shape, "n_heads", shape_overrides.pop("n_heads", 12)) or 12)
    n_ffn = int(_shape_get(shape, "n_ffn", shape_overrides.pop("n_ffn", d_model * 4)) or d_model * 4)
    n_layers = int(_shape_get(shape, "n_layers", shape_overrides.pop("n_layers", 12)) or 12)
    ffn_linear_count = int(_shape_get(shape, "ffn_linear_count", shape_overrides.pop("ffn_linear_count", 2)) or 2)
    attn_linear_count = int(_shape_get(shape, "attn_linear_count", shape_overrides.pop("attn_linear_count", 4)) or 4)

    block_scores, dim_scores = block_scores_from_fisher(scores, d_model=d_model, verbose=verbose)

    selector = shape_overrides.pop("selector", None)
    if selector is None and shape is not None:
        try:
            from apt.block_selection import BlockSelector

            selector = BlockSelector(shape)
        except Exception:
            selector = None

    if selector is not None:
        try:
            blocks = None
            for attempt in (lambda: selector.enumerate_blocks(), lambda: selector.enumerate_blocks({})):
                try:
                    blocks = attempt()
                    break
                except TypeError:
                    continue
            if blocks:
                for block in blocks:
                    kind = int(getattr(block, "kind", HEAD))
                    layer = int(getattr(block, "layer", -1))
                    index = int(getattr(block, "index", -1))
                    if kind == DIMENSION:
                        salience = float(dim_scores.get(index, 0.0))
                    else:
                        site = getattr(block, "site", None)
                        salience = block_scores.get((kind, layer, index, site))
                        if salience is None:
                            salience = block_scores.get((kind, layer, index, None), 0.0)
                    try:
                        block.salience = float(salience)
                    except Exception:
                        pass
                try:
                    selector.recompute_densities(blocks)
                except Exception:
                    pass
                selection = None
                for attempt in (
                    lambda: selector.select_to_sparsity(blocks, float(sparsity)),
                    lambda: selector.select_to_sparsity(blocks, float(sparsity), None, None),
                ):
                    try:
                        selection = attempt()
                        break
                    except TypeError:
                        continue
                if selection is not None:
                    if verbose:
                        print(f"[mask_tuning] block selector retained {len(getattr(selection, 'retained', []) or [])} blocks")
                    return selection
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"[mask_tuning] block selector unavailable ({exc}); using greedy fallback")

    selection = _fallback_select(
        block_scores,
        dim_scores,
        d_model=d_model,
        n_heads=n_heads,
        n_ffn=n_ffn,
        n_layers=n_layers,
        ffn_linear_count=ffn_linear_count,
        attn_linear_count=attn_linear_count,
        sparsity=float(sparsity),
        prune_heads=prune_heads,
        prune_neurons=prune_neurons,
        prune_dims=prune_dims,
    )
    if verbose:
        print(
            f"[mask_tuning] greedy selection kept {len(selection.retained)} blocks "
            f"(target sparsity {sparsity:.2f}, realised {selection.sparsity():.3f})"
        )
    return selection


# ---------------------------------------------------------------------------
# mask application + physical pruning
# ---------------------------------------------------------------------------


def _retained_indices(selection: Any) -> Dict[str, Any]:
    from collections import defaultdict

    heads: Dict[int, set] = defaultdict(set)
    neurons: Dict[int, set] = defaultdict(set)
    dims: set = set()
    for block in getattr(selection, "retained", []) or []:
        kind = int(getattr(block, "kind", HEAD))
        layer = int(getattr(block, "layer", -1))
        index = int(getattr(block, "index", -1))
        if kind == HEAD:
            heads[layer].add(index)
        elif kind == NEURON:
            neurons[layer].add(index)
        else:
            dims.add(index)
    return {"heads": heads, "neurons": neurons, "dims": dims}


def _finish_masks(model: Any, threshold: float = DEFAULT_MASK_THRESHOLD) -> None:
    """Push masks into the model and snap them to 0/1."""
    for name in ("sync_masks", "apply_masks"):
        method = getattr(model, name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    method = getattr(model, "harden_masks", None)
    if callable(method):
        try:
            method(float(threshold))
        except TypeError:
            try:
                method()
            except Exception:
                pass
        except Exception:
            pass


def apply_binary_masks(
    model: Any,
    selection: Any,
    *,
    shape: Any = None,
    threshold: float = DEFAULT_MASK_THRESHOLD,
    verbose: bool = False,
) -> bool:
    """Push hard 0/1 masks derived from ``selection`` onto the wrapped model."""
    torch = _torch()

    apply_selection = getattr(model, "apply_selection", None)
    if callable(apply_selection):
        try:
            apply_selection(selection)
            _finish_masks(model, threshold)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                warnings.warn(f"[mask_tuning] wrapper.apply_selection failed ({exc}); setting masks manually")

    kept = _retained_indices(selection)
    if shape is None:
        try:
            from apt.model_wrapper import apt_shape

            shape = apt_shape(model)
        except Exception:
            shape = None
    d_model = int(_shape_get(shape, "d_model", 0) or 0)
    n_heads = int(_shape_get(shape, "n_heads", 0) or 0)
    n_ffn = int(_shape_get(shape, "n_ffn", 0) or 0)
    n_layers = int(_shape_get(shape, "n_layers", 0) or 0)
    if d_model <= 0 or n_heads <= 0 or n_ffn <= 0 or n_layers <= 0:
        raise RuntimeError("cannot determine the model shape required to build the pruning masks")

    dim_mask = torch.zeros(int(d_model))
    for index in kept["dims"]:
        if 0 <= int(index) < d_model:
            dim_mask[int(index)] = 1.0
    applied = 0
    set_dim_mask = getattr(model, "set_dim_mask", None)
    if callable(set_dim_mask):
        try:
            set_dim_mask(dim_mask)
            applied += 1
        except Exception as exc:  # pragma: no cover
            if verbose:
                warnings.warn(f"[mask_tuning] set_dim_mask failed: {exc}")

    for layer in range(int(n_layers)):
        head_mask = torch.zeros(int(n_heads))
        for index in kept["heads"].get(layer, set()):
            if 0 <= int(index) < n_heads:
                head_mask[int(index)] = 1.0
        neuron_mask = torch.zeros(int(n_ffn))
        for index in kept["neurons"].get(layer, set()):
            if 0 <= int(index) < n_ffn:
                neuron_mask[int(index)] = 1.0
        for name, mask in (("set_head_mask", head_mask), ("set_neuron_mask", neuron_mask)):
            method = getattr(model, name, None)
            if not callable(method):
                continue
            try:
                method(layer, mask)
                applied += 1
            except TypeError:
                try:
                    method(mask)
                    applied += 1
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover
                if verbose:
                    warnings.warn(f"[mask_tuning] {name}({layer}) failed: {exc}")

    _finish_masks(model, threshold)
    return applied > 0


def physical_prune(model: Any, *, threshold: float = DEFAULT_MASK_THRESHOLD) -> Any:
    """Merge tuning weights and remove pruned heads/neurons/dims (plain HF model)."""
    try:
        from apt.merge import merge_and_prune

        return merge_and_prune(model, threshold=float(threshold))
    except Exception as exc:
        warnings.warn(f"[mask_tuning] apt.merge.merge_and_prune unavailable ({exc}); returning the wrapped model")
        return model


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def _accuracy(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    n = min(len(predictions), len(references))
    if n == 0:
        return float("nan")
    correct = sum(int(round(float(p)) == int(round(float(r)))) for p, r in zip(predictions[:n], references[:n]))
    return 100.0 * correct / float(n)


def _primary_from_metrics(task: str, metrics: Dict[str, float]) -> float:
    if not metrics:
        return float("nan")
    try:
        from apt.eval.metrics import primary_metric

        value = primary_metric(canonical_task(task), metrics)
        if value is not None:
            return float(value)
    except Exception:
        pass
    for key in ("accuracy", "f1", "matthews_correlation", "spearmanr", "exact", "rougeL", "rouge_l"):
        if key in metrics:
            return float(metrics[key])
    return float(next(iter(metrics.values())))


def _evaluate_model(
    model: Any,
    dataloader: Any,
    *,
    task: str = "sst2",
    tokenizer: Any = None,
    device: Any = None,
    model_name: str = "",
    sparsity: float = 0.0,
    method: str = "mask_tuning",
    max_batches: Optional[int] = None,
    compute_metrics: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, float], float]:
    """Evaluate ``model``; returns ``(metrics_dict, primary_metric)``."""
    task = canonical_task(task)
    if dataloader is None:
        return {}, float("nan")

    try:
        from apt.eval.run_eval import evaluate_model as harness_evaluate

        result = harness_evaluate(
            model,
            dataloader,
            task=task,
            tokenizer=tokenizer,
            device=device,
            method=method,
            model_name=model_name,
            sparsity=float(sparsity),
            max_batches=max_batches,
            compute_efficiency=False,
        )
        metrics = dict(getattr(result, "metrics", {}) or {})
        primary = getattr(result, "primary", None)
        return metrics, float(primary if primary is not None else _primary_from_metrics(task, metrics))
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(f"[mask_tuning] apt.eval harness unavailable ({exc}); using the local evaluation loop")

    torch = _torch()
    device = resolve_device(device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    predictions: List[Any] = []
    references: List[Any] = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= int(max_batches):
                break
            inputs = model_inputs(batch)
            labels = inputs.pop("labels", None)
            if labels is None:
                labels = batch.get("labels") if isinstance(batch, dict) else None
            inputs = move_to_device(inputs, device)
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                logits = outputs[0]
            if logits is None:
                continue
            logits = logits.detach().float().cpu()
            if logits.dim() > 1 and logits.shape[-1] > 1:
                predictions.extend(logits.argmax(dim=-1).reshape(-1).tolist())
            else:
                predictions.extend(logits.reshape(-1).tolist())
            if labels is not None:
                references.extend(_to_list(labels))
    if was_training:
        model.train()
    if not predictions:
        return {}, float("nan")
    try:
        from apt.eval.metrics import compute_metrics as metrics_dispatch

        metrics = dict(metrics_dispatch(task, predictions, references))
    except Exception:
        if compute_metrics is not None:
            try:
                metrics = dict(compute_metrics(predictions, references))
            except Exception:
                metrics = {"accuracy": _accuracy(predictions, references)}
        else:
            metrics = {"accuracy": _accuracy(predictions, references)}
    return metrics, float(_primary_from_metrics(task, metrics))


def _lr_factor(step: int, total_steps: int, *, warmup_steps: int = 0, kind: str = "linear") -> float:
    try:
        from apt.baselines.ft import lr_factor

        return float(lr_factor(step, total_steps, warmup_steps=warmup_steps, kind=kind))
    except Exception:
        pass
    total_steps = max(1, int(total_steps))
    step = int(step)
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.0, 1.0 - progress)


def _interpolate_tta(prev_entry: Optional[Dict[str, Any]], entry: Dict[str, Any], target: float) -> float:
    """Linearly interpolate the wall-clock time at which ``target`` was reached."""
    if prev_entry is None:
        return float(entry["elapsed_s"])
    prev_primary = _primary_from_metrics("", prev_entry.get("metrics", {}) or {})
    curr_primary = _primary_from_metrics("", entry.get("metrics", {}) or {})
    if math.isnan(prev_primary) or math.isnan(curr_primary) or curr_primary == prev_primary:
        return float(entry["elapsed_s"])
    frac = min(1.0, max(0.0, (target - prev_primary) / (curr_primary - prev_primary)))
    return float(prev_entry["elapsed_s"] + frac * (entry["elapsed_s"] - prev_entry["elapsed_s"]))


def _train_loop(
    model: Any,
    train_dataloader: Any,
    *,
    epochs: int,
    config: MaskTuningConfig,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable[..., Any]] = None,
    tokenizer: Any = None,
    reference_metric: Optional[float] = None,
    device: Any = None,
    task: Optional[str] = None,
    method: str = "mask_tuning",
    sparsity: float = 0.0,
    model_name: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train the currently-trainable parameters (LoRA adapters) of ``model``."""
    torch = _torch()
    device = resolve_device(device)
    task = canonical_task(task or config.task)
    if device is not None:
        model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:  # everything frozen -> train all (safety)
        trainable = list(model.parameters())
        for param in trainable:
            param.requires_grad_(True)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(config.weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
    )

    steps_per_epoch = max(1, len(train_dataloader) if hasattr(train_dataloader, "__len__") else 1)
    if config.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, int(config.max_train_batches))
    total_steps = max(1, steps_per_epoch * int(epochs))
    warmup_steps = int(round(float(config.warmup_ratio) * total_steps))

    history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    tta: Dict[str, Any] = {"seconds": None, "target": None, "value": None}
    if reference_metric is not None and not math.isnan(float(reference_metric)):
        tta["target"] = float(TTA_FRACTION) * float(reference_metric)

    if device is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    global_step = 0
    last_metrics: Dict[str, float] = {}
    for epoch in range(int(epochs)):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_dataloader):
            if config.max_train_batches and step >= int(config.max_train_batches):
                break
            inputs = move_to_device(model_inputs(batch), device)
            outputs = model(**inputs)
            loss = _task_loss(outputs, inputs)
            if loss is None:
                continue
            loss.backward()
            if config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(config.max_grad_norm)
                )
            factor = _lr_factor(global_step, total_steps, warmup_steps=warmup_steps, kind=str(config.lr_kind))
            for group in optimizer.param_groups:
                group["lr"] = float(config.learning_rate) * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item())
            seen += 1
            global_step += 1
            if verbose and config.logging_steps and global_step % int(config.logging_steps) == 0:
                print(f"[mask_tuning] epoch {epoch} step {global_step} loss {running / max(1, seen):.4f}")
        epoch_loss = running / max(1, seen)
        history.append(
            {"epoch": epoch, "step": global_step, "loss": epoch_loss, "elapsed_s": time.time() - start}
        )
        if verbose:
            print(f"[mask_tuning] epoch {epoch} done, mean loss {epoch_loss:.4f}")

        if eval_dataloader is not None:
            metrics, primary = _evaluate_model(
                model,
                eval_dataloader,
                task=task,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                sparsity=sparsity,
                method=method,
                max_batches=config.max_eval_batches,
                compute_metrics=compute_metrics,
            )
            if metrics:
                last_metrics = metrics
                eval_entry = {
                    "epoch": epoch,
                    "step": global_step,
                    "elapsed_s": time.time() - start,
                    "metrics": metrics,
                }
                eval_history.append(eval_entry)
                if verbose:
                    print(f"[mask_tuning] eval@epoch {epoch}: {metrics}")
                if tta["seconds"] is None and tta["target"] is not None and not math.isnan(primary):
                    if primary >= tta["target"]:
                        prev = eval_history[-2] if len(eval_history) > 1 else None
                        tta["seconds"] = _interpolate_tta(prev, eval_entry, tta["target"])
                        tta["value"] = primary

    train_time = time.time() - start
    peak_mb = None
    if device is not None and torch.cuda.is_available():
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
    if eval_dataloader is not None and not last_metrics:
        last_metrics, _ = _evaluate_model(
            model,
            eval_dataloader,
            task=task,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            sparsity=sparsity,
            method=method,
            max_batches=config.max_eval_batches,
            compute_metrics=compute_metrics,
        )
    return {
        "train_time_s": float(train_time),
        "train_peak_mem_mb": peak_mb,
        "tta_seconds": tta["seconds"],
        "tta_target": tta["target"],
        "history": history,
        "eval_history": eval_history,
        "metrics": last_metrics,
        "num_steps": global_step,
    }


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class MaskTuningPruneResult:
    model: Any
    selection: Any = None
    scores: Any = None
    sparsity: float = 0.0
    num_parameters: Optional[int] = None
    removed: Dict[str, Any] = field(default_factory=dict)
    scoring: str = "fisher"
    n_batches: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "sparsity": float(self.sparsity),
            "num_parameters": self.num_parameters,
            "removed": dict(self.removed),
            "scoring": self.scoring,
            "fisher_batches": self.n_batches,
        }


def _count_parameters(model: Any) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------


class MaskTuningMethod:
    """End-to-end ``LoRA+Prune`` (Mask Tuning) baseline.

    Stage 1 LoRA tuning -> Stage 2 Fisher-based structured pruning (heads, FFN
    neurons, hidden dimensions) -> Stage 3 post-pruning retraining.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Any = None,
        tokenizer: Any = None,
        train_dataloader: Any = None,
        eval_dataloader: Any = None,
        compute_metrics: Optional[Callable[..., Any]] = None,
        reference_metric: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = MaskTuningConfig.from_dict(kwargs.pop("config_dict", None), **kwargs)
        elif isinstance(config, dict):
            config = MaskTuningConfig.from_dict(config, **kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
        self.config = apply_table6_defaults(config)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.reference_metric = reference_metric

        self.device = resolve_device(self.config.device)
        self.wrapped_model: Any = None
        self.pruned_model: Any = None
        self.selection: Any = None
        self.scores: Any = None
        self.stage_records: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.primary: Optional[float] = None
        self.num_parameters: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        self._external_module: Optional[ModuleType] = None
        set_seed(self.config.seed)

    # -- setup -----------------------------------------------------------
    def setup_model(self, *, build: bool = True) -> Any:
        if self.model is None and build:
            self.model, self.tokenizer = build_model_and_tokenizer(self.config)
        if self.model is not None and self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, use_fast=True)
        if self.model is not None and self.device is not None:
            self.model.to(self.device)
        return self.model

    def setup_data(self, splits: Sequence[str] = ("train", "validation")) -> Dict[str, Any]:
        if self.train_dataloader is not None and self.eval_dataloader is not None:
            return {"train": self.train_dataloader, "validation": self.eval_dataloader}
        self.setup_model()
        loaders = build_dataloaders_for_task(self.config, self.tokenizer, splits=splits)
        self.train_dataloader = loaders.get("train", self.train_dataloader)
        self.eval_dataloader = loaders.get("validation", loaders.get("dev", self.eval_dataloader))
        return loaders

    def _load_external(self) -> Optional[ModuleType]:
        if not self.config.use_external_repo:
            return None
        if self._external_module is None:
            self._external_module = load_external_repo(self.config.external_repo_path)
            if self._external_module is None:
                warnings.warn(
                    "Mask Tuning repository not found; using the in-repo Fisher-information "
                    f"pruning adaptation (clone {MASK_TUNING_REPO_URL} or set MASK_TUNING_REPO)."
                )
        return self._external_module

    # -- stage 1 ---------------------------------------------------------
    def wrap_for_tuning(self) -> Any:
        """Wrap the LM with trainable LoRA-style adapters and salience caching."""
        from apt.model_wrapper import wrap_model

        model_type = self.config.model_type or self.config.model_name_or_path
        self.wrapped_model = wrap_model(
            self.model,
            rank=int(self.config.lora_rank),
            scaling=float(self.config.resolved_scaling),
            model_type=model_type,
            wrap_attn=True,
            wrap_ffn=bool(self.config.tune_ffn_adapters),
            wrap_cross_attn=True,
            wrap_mask_only=False,
            cache_for_salience=True,
            capture_grad=True,
        )
        if self.device is not None:
            self.wrapped_model.to(self.device)
        return self.wrapped_model

    def tune_lora(self, *, verbose: bool = True) -> Dict[str, Any]:
        """Stage 1: LoRA-style tuning of the unpruned LM (pruning is post-hoc)."""
        self.setup_model()
        if self.wrapped_model is None:
            self.wrap_for_tuning()
        record = _train_loop(
            self.wrapped_model,
            self.train_dataloader,
            epochs=int(self.config.lora_epochs),
            config=self.config,
            eval_dataloader=None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            device=self.device,
            task=self.config.task,
            method="mask_tuning/lora",
            verbose=verbose,
        )
        self.stage_records["lora"] = record
        self.history.extend(record.get("history", []))
        return record

    # -- stage 2 ---------------------------------------------------------
    def score_fisher(self, *, verbose: bool = True) -> FisherScoreSet:
        """Accumulate Fisher information of the mask variables over the data."""
        if self.wrapped_model is None:
            self.wrap_for_tuning()
        try:
            from apt.model_wrapper import enable_grad_capture, enable_salience_cache

            enable_salience_cache(self.wrapped_model, True)
            enable_grad_capture(self.wrapped_model, True)
        except Exception:
            pass
        self.scores = compute_fisher_scores(
            self.wrapped_model,
            self.train_dataloader,
            task=self.config.task,
            device=self.device,
            num_batches=int(self.config.fisher_num_batches),
            pool=str(self.config.fisher_pool),
            scoring=str(self.config.scoring),
            max_batches=self.config.max_train_batches,
            verbose=verbose,
        )
        return self.scores

    def select(self, *, verbose: bool = True):
        if self.scores is None:
            self.score_fisher(verbose=verbose)
        self.selection = select_blocks_from_fisher(
            self.wrapped_model,
            self.scores,
            sparsity=float(self.config.resolved_sparsity),
            prune_heads=bool(self.config.prune_heads),
            prune_neurons=bool(self.config.prune_neurons),
            prune_dims=bool(self.config.prune_dims),
            verbose=verbose,
        )
        return self.selection

    def prune(self, *, verbose: bool = True) -> Any:
        """Apply pruned masks, then physically remove them from the LM."""
        if self.selection is None:
            self.select(verbose=verbose)
        applied = apply_binary_masks(
            self.wrapped_model, self.selection, threshold=float(self.config.mask_threshold), verbose=verbose
        )
        if not applied and verbose:
            warnings.warn("[mask_tuning] no masks were applied through the wrapper API")
        self.pruned_model = physical_prune(self.wrapped_model, threshold=float(self.config.mask_threshold))
        self.num_parameters = _count_parameters(self.pruned_model)
        return self.pruned_model

    # -- stage 3 ---------------------------------------------------------
    def recover(self, *, verbose: bool = True) -> Dict[str, Any]:
        """Stage 3: retrain the pruned (physically smaller) LM to recover accuracy."""
        if self.pruned_model is None:
            self.prune(verbose=verbose)
        model = self.pruned_model
        try:
            from apt.baselines.lora import apply_lora, lora_parameter_count, merge_lora

            if lora_parameter_count(model, trainable_only=False) == 0:
                apply_lora(
                    model,
                    rank=int(self.config.lora_rank),
                    scaling=float(self.config.resolved_scaling),
                    dropout=float(self.config.lora_dropout),
                    target_modules=self.config.resolved_target_modules,
                    model_type=self.config.model_type or self.config.model_name_or_path,
                )
            self._merge_lora_after = merge_lora
        except Exception as exc:  # pragma: no cover - LoRA baseline optional
            warnings.warn(f"[mask_tuning] LoRA recovery unavailable ({exc}); retraining all parameters")
            self._merge_lora_after = None
        record = _train_loop(
            model,
            self.train_dataloader,
            epochs=int(self.config.recovery_epochs),
            config=self.config,
            eval_dataloader=self.eval_dataloader,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            reference_metric=self.reference_metric,
            device=self.device,
            task=self.config.task,
            method="mask_tuning/recovery",
            sparsity=float(self.config.resolved_sparsity),
            model_name=self.config.model_name_or_path,
            verbose=verbose,
        )
        self.stage_records["recovery"] = record
        self.history.extend(record.get("history", []))
        self.metrics = dict(record.get("metrics", {}) or {})
        self.primary = _primary_from_metrics(self.config.task, self.metrics) if self.metrics else None
        return record

    # -- top level -------------------------------------------------------
    def fit(self, *, verbose: bool = True, evaluate: bool = True) -> Dict[str, Any]:
        self.setup_data()
        if verbose:
            self._load_external()
        self.tune_lora(verbose=verbose)
        self.score_fisher(verbose=verbose)
        self.select(verbose=verbose)
        self.prune(verbose=verbose)
        record = self.recover(verbose=verbose)
        if evaluate and not self.metrics:
            self.metrics, self.primary = _evaluate_model(
                self.pruned_model,
                self.eval_dataloader,
                task=self.config.task,
                tokenizer=self.tokenizer,
                device=self.device,
                model_name=self.config.model_name_or_path,
                sparsity=float(self.config.resolved_sparsity),
                method="mask_tuning",
                max_batches=self.config.max_eval_batches,
                compute_metrics=self.compute_metrics,
            )
        return self.summary(record)

    def evaluate(self, dataloader: Any = None, *, merge_tuning: bool = True) -> Dict[str, float]:
        """Evaluate the pruned model (optionally merging the recovery LoRA first)."""
        model = self.pruned_model if self.pruned_model is not None else self.model
        if merge_tuning:
            self._merge_recovery_lora()
        self.metrics, self.primary = _evaluate_model(
            model,
            dataloader if dataloader is not None else self.eval_dataloader,
            task=self.config.task,
            tokenizer=self.tokenizer,
            device=self.device,
            model_name=self.config.model_name_or_path,
            sparsity=float(self.config.resolved_sparsity),
            method="mask_tuning",
            max_batches=self.config.max_eval_batches,
            compute_metrics=self.compute_metrics,
        )
        return self.metrics

    def _merge_recovery_lora(self) -> None:
        merge = getattr(self, "_merge_lora_after", None)
        if merge is None:
            return
        try:
            merge(self.pruned_model)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"[mask_tuning] merging the recovery LoRA failed: {exc}")

    def prune_summary(self) -> Dict[str, Any]:
        if self.selection is None:
            return {}
        kept = _retained_indices(self.selection)
        return {
            "kept_heads_per_layer": {int(k): len(v) for k, v in kept["heads"].items()},
            "kept_neurons_per_layer": {int(k): len(v) for k, v in kept["neurons"].items()},
            "kept_dimensions": len(kept["dims"]),
            "n_retained_blocks": len(getattr(self.selection, "retained", []) or []),
            "n_pruned_blocks": len(getattr(self.selection, "pruned", []) or []),
        }

    def summary(self, record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        record = record or {}
        lora_record = self.stage_records.get("lora", {})
        recovery_record = self.stage_records.get("recovery", record)
        train_time = float(lora_record.get("train_time_s", 0.0) or 0.0) + float(
            recovery_record.get("train_time_s", 0.0) or 0.0
        )
        peaks = [
            value
            for value in (lora_record.get("train_peak_mem_mb"), recovery_record.get("train_peak_mem_mb"))
            if value is not None
        ]
        return {
            "method": "mask_tuning",
            "display_name": "LoRA+Prune",
            "model": self.config.model_name_or_path,
            "task": canonical_task(self.config.task),
            "sparsity": float(self.config.resolved_sparsity),
            "metrics": dict(self.metrics),
            "primary": self.primary,
            "train_time_s": train_time,
            "train_peak_mem_mb": max(peaks) if peaks else None,
            "tta_seconds": recovery_record.get("tta_seconds", lora_record.get("tta_seconds")),
            "num_parameters": self.num_parameters,
            "pruning": self.prune_summary(),
            "stages": {
                "lora_epochs": int(self.config.lora_epochs),
                "recovery_epochs": int(self.config.recovery_epochs),
                "lora_time_s": lora_record.get("train_time_s"),
                "recovery_time_s": recovery_record.get("train_time_s"),
            },
            "scoring": {
                "mode": getattr(self.scores, "scoring", str(self.config.scoring)),
                "fisher_batches": getattr(self.scores, "n_batches", 0),
            },
            "external_repo": {
                "url": MASK_TUNING_REPO_URL,
                "path": locate_external_repo(self.config.external_repo_path),
                "available": bool(external_repo_available(self.config.external_repo_path)),
            },
            "history": self.history,
        }

    def save(self, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "mask_tuning_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, default=str)
        if self.pruned_model is not None:
            try:
                self.pruned_model.save_pretrained(output_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"[mask_tuning] could not save the pruned model: {exc}")
        return path


# ---------------------------------------------------------------------------
# functional entry points
# ---------------------------------------------------------------------------


def prune_with_mask_tuning(
    model: Any,
    dataloader: Any = None,
    *,
    task: str = "sst2",
    tokenizer: Any = None,
    sparsity: float = DEFAULT_TARGET_SPARSITY,
    device: Any = None,
    num_batches: Optional[int] = DEFAULT_FISHER_NUM_BATCHES,
    pool: str = "token",
    scoring: str = "auto",
    threshold: float = DEFAULT_MASK_THRESHOLD,
    prune_heads: bool = True,
    prune_neurons: bool = True,
    prune_dims: bool = True,
    verbose: bool = True,
) -> MaskTuningPruneResult:
    """Run Mask Tuning's Fisher-based structured pruning on a LoRA-tuned LM.

    The model is wrapped with :func:`apt.model_wrapper.wrap_model` (mask-only,
    numerically transparent), Fisher information is accumulated over
    ``dataloader``, blocks are ranked by density under the sparsity constraint,
    masks are set, and pruned heads / neurons / dimensions are physically removed.
    """
    from apt.model_wrapper import is_wrapped, wrap_model

    try:
        wrapped = model if is_wrapped(model) else wrap_model(
            model, rank=0, scaling=1.0, wrap_mask_only=True, cache_for_salience=True, capture_grad=True
        )
    except Exception:
        wrapped = wrap_model(model, wrap_mask_only=True, cache_for_salience=True, capture_grad=True)
    if device is not None:
        wrapped.to(resolve_device(device))

    scores = compute_fisher_scores(
        wrapped,
        dataloader,
        task=task,
        device=device,
        num_batches=num_batches,
        pool=pool,
        scoring=scoring,
        verbose=verbose,
    )
    selection = select_blocks_from_fisher(
        wrapped,
        scores,
        sparsity=float(sparsity),
        prune_heads=prune_heads,
        prune_neurons=prune_neurons,
        prune_dims=prune_dims,
        verbose=verbose,
    )
    apply_binary_masks(wrapped, selection, threshold=threshold, verbose=verbose)
    pruned = physical_prune(wrapped, threshold=threshold)
    kept = _retained_indices(selection)
    removed = {
        "kept_heads": sum(len(v) for v in kept["heads"].values()),
        "kept_neurons": sum(len(v) for v in kept["neurons"].values()),
        "kept_dimensions": len(kept["dims"]),
        "n_pruned_blocks": len(getattr(selection, "pruned", []) or []),
    }
    return MaskTuningPruneResult(
        model=pruned,
        selection=selection,
        scores=scores,
        sparsity=float(sparsity),
        num_parameters=_count_parameters(pruned),
        removed=removed,
        scoring=getattr(scores, "scoring", scoring),
        n_batches=int(getattr(scores, "n_batches", 0) or 0),
    )


def train_mask_tuning(
    config: Optional[Any] = None,
    model: Any = None,
    tokenizer: Any = None,
    train_dataloader: Any = None,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable[..., Any]] = None,
    reference_metric: Optional[float] = None,
    *,
    verbose: bool = True,
    evaluate: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-call ``LoRA+Prune`` baseline (the Table 2 ``LoRA+Prune`` row)."""
    method = MaskTuningMethod(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics,
        reference_metric=reference_metric,
        **kwargs,
    )
    summary = method.fit(verbose=verbose, evaluate=evaluate)
    summary["trainer"] = method
    return summary


def evaluate_mask_tuning(
    source: Any,
    dataloader: Any = None,
    *,
    task: str = "sst2",
    tokenizer: Any = None,
    device: Any = None,
    compute_metrics: Optional[Callable[..., Any]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a :class:`MaskTuningMethod` or a plain model."""
    if isinstance(source, MaskTuningMethod):
        metrics = source.evaluate(dataloader)
        summary = source.summary()
        summary["metrics"] = metrics
    else:
        metrics, primary = _evaluate_model(
            source,
            dataloader,
            task=task,
            tokenizer=tokenizer,
            device=device,
            method="mask_tuning",
            compute_metrics=compute_metrics,
        )
        summary = {"method": "mask_tuning", "task": canonical_task(task), "metrics": metrics, "primary": primary}
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
    return summary


#: alias used by :mod:`apt.baselines` registry code
mask_tuning_trainer = MaskTuningMethod


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    ok = True

    # 1) Appendix C parameter-count estimates on RoBERTa-base shapes.
    head_cost, neuron_cost, dim_cost = _block_costs(768, 12, 3072, 12, 2, 4)
    for name, value, expected in (
        ("head", head_cost, 196608.0),
        ("neuron", neuron_cost, 1536.0),
        ("dimension", dim_cost, 110592.0),
    ):
        if abs(float(value) - expected) > 1e-6 * expected:
            print(f"[mask_tuning] FAIL C_{name}: {value} != {expected}")
            ok = False

    # 2) config resolution / Table 6 defaults
    config = MaskTuningConfig.from_dict({"task": "mnli", "target_sparsity": 0.6})
    apply_table6_defaults(config)
    if config.table6_group != "glue-big" or config.learning_rate != 2.0e-4 or config.batch_size != 32:
        print(f"[mask_tuning] FAIL config: {config.table6_group}/{config.learning_rate}/{config.batch_size}")
        ok = False
    if config.lora_epochs != 20 or config.recovery_epochs != 20:
        print(f"[mask_tuning] FAIL epochs: {config.lora_epochs}/{config.recovery_epochs}")
        ok = False
    if abs(config.resolved_sparsity - 0.6) > 1e-9 or abs(config.resolved_scaling - 2.0) > 1e-9:
        print(f"[mask_tuning] FAIL sparsity/scaling: {config.resolved_sparsity}/{config.resolved_scaling}")
        ok = False
    cnn = MaskTuningConfig.from_dict({"task": "cnndm"})
    apply_table6_defaults(cnn)
    if cnn.batch_size != 16 or cnn.epochs != 16 or cnn.max_seq_length != 512 or cnn.table6_group != "cnndm":
        print(f"[mask_tuning] FAIL cnndm: {cnn.batch_size}/{cnn.epochs}/{cnn.max_seq_length}/{cnn.table6_group}")
        ok = False

    # 3) task helpers
    if canonical_task("SST-2") != "sst2" or not is_glue_task("MNLI") or not is_squad_task("squad_v2"):
        print("[mask_tuning] FAIL task normalisation")
        ok = False
    if not is_seq2seq_task("cnn_dailymail") or canonical_task("cnn_dailymail") != "cnndm":
        print("[mask_tuning] FAIL seq2seq detection")
        ok = False

    # 4) greedy density selection honours the sparsity budget (synthetic scores)
    block_scores: Dict[Tuple[int, int, int, Optional[str]], float] = {}
    for layer in range(2):
        for head in range(4):
            block_scores[(HEAD, layer, head, None)] = float(head + 1)
        for neuron in range(8):
            block_scores[(NEURON, layer, neuron, None)] = float(neuron + 1)
    dim_scores = {i: float(i + 1) for i in range(8)}
    selection = _fallback_select(
        block_scores,
        dim_scores,
        d_model=8,
        n_heads=4,
        n_ffn=8,
        n_layers=2,
        ffn_linear_count=2,
        attn_linear_count=4,
        sparsity=0.5,
    )
    realised = selection.sparsity()
    if not (0.0 <= realised <= 1.0) or len(selection.pruned) == 0 or len(selection.retained) == 0:
        print(f"[mask_tuning] FAIL selection: sparsity={realised}, pruned={len(selection.pruned)}")
        ok = False
    if selection.param_count > selection.original_param_count:
        print("[mask_tuning] FAIL retained params exceed the original count")
        ok = False
    # the highest-density block must survive
    top_kind = max(selection.retained, key=lambda b: b.density)
    if top_kind.density < max(b.density for b in selection.pruned):
        print("[mask_tuning] FAIL density ordering")
        ok = False

    # 5) external repo detection is side-effect free
    path = locate_external_repo()
    if path is not None and not _looks_like_repo(path):
        print(f"[mask_tuning] FAIL repo detection: {path}")
        ok = False

    if ok:
        print(f"[mask_tuning] all self-tests passed (external repo: {'found' if path else 'not found'})")
    return ok


if __name__ == "__main__":  # pragma: no cover
    _self_test()
