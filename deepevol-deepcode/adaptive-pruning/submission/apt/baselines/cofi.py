"""CoFi baseline and the LoRA+Prune+Distill variant (paper Sec. 5.2).

Paper description (Sec. 5.2 "Baselines"):

* **Prune+Distill** -- "knowledge distillation has been proved to be a key
  technique in recovering pruned LMs' task accuracy.  In particular, we use
  the state-of-the-art pruning plus distillation method called CoFi (Xia et
  al., 2022) which uses :math:`L_0` regularization for pruning plus dynamic
  layer-wise distillation objectives.  We only compare APT to CoFi with
  RoBERTa models since the training memory usage of CoFi is too high for
  larger LMs."
* **LoRA+Prune+Distill** -- "to reduce the training memory consumption in
  pruning and distillation, a simple baseline is to conduct CoFi pruning and
  distillation but with LoRA parameters tuned only.  More specifically, only
  the :math:`L_0` module and LoRA parameters are tunable under this setting."

The official implementation lives at
``https://github.com/princeton-nlp/CoFiPruning`` and is used with its default
hyper-parameters (learning rate, batch size, :math:`L_0` coefficients, layer
distillation weights, temperature, ...).  This module therefore

1. locates / imports / (optionally) clones that external repository, and
2. provides a fully in-repo, dependency-light implementation of the same
   recipe (hard-concrete :math:`L_0` gates over structured units + :math:`L_0`
   regularization + dynamic layer-wise hidden-state distillation) which is used
   whenever the external checkout is unavailable.

The reference numbers this file is validated against (Table 2, RoBERTa-base,
60% sparsity, MNLI/SST2): ``Prune+Distill`` = 87.3 / 94.5 with Train Time
1495.3%, Train Mem 168.5%, and ``LoRA+Prune+Distill`` = 84.2 / 91.9 with
Train Time 6534.6%, Train Mem 141.4%.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# cross-module imports (all optional: the module must import in a bare env)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - trivial
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _torch():
    """Return the ``torch`` module, importing lazily and raising if absent."""
    global torch, nn
    if torch is None:  # pragma: no cover - only without torch
        try:
            import torch as _t  # noqa: WPS433
            import torch.nn as _nn  # noqa: WPS433

            torch = _t
            nn = _nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for the CoFi baseline but could not be imported"
            ) from exc
    return torch


try:
    from apt.adapters import DIMENSION, HEAD, NEURON, iter_masked_linears
except Exception:  # pragma: no cover - defensive fallback
    HEAD, NEURON, DIMENSION = 0, 1, 2

    def iter_masked_linears(module, names=None):  # type: ignore
        for name, sub in module.named_modules():
            if hasattr(sub, "base_weight") and hasattr(sub, "adapter"):
                if names is None or any(n in name for n in names):
                    yield name, sub

try:
    from apt.block_selection import (
        BlockSelector,
        ModelShape,
        dimension_param_count,
        head_param_count,
        neuron_param_count,
    )
except Exception:  # pragma: no cover
    BlockSelector = None  # type: ignore
    ModelShape = None  # type: ignore
    head_param_count = None  # type: ignore
    neuron_param_count = None  # type: ignore
    dimension_param_count = None  # type: ignore

try:
    from apt.model_wrapper import (
        apt_shape,
        enable_grad_capture,
        enable_salience_cache,
        is_wrapped,
        wrap_model,
    )
except Exception:  # pragma: no cover
    apt_shape = None  # type: ignore
    enable_grad_capture = None  # type: ignore
    enable_salience_cache = None  # type: ignore
    is_wrapped = None  # type: ignore
    wrap_model = None  # type: ignore

try:
    from apt.merge import merge_and_prune
except Exception:  # pragma: no cover
    merge_and_prune = None  # type: ignore

try:
    from apt.distillation import find_transformer_layers
except Exception:  # pragma: no cover
    find_transformer_layers = None  # type: ignore

try:
    from apt.eval.run_eval import evaluate_model
except Exception:  # pragma: no cover
    evaluate_model = None  # type: ignore

try:
    from apt.eval.metrics import compute_metrics as _compute_metrics
    from apt.eval.metrics import primary_metric as _primary_metric
except Exception:  # pragma: no cover
    _compute_metrics = None  # type: ignore
    _primary_metric = None  # type: ignore

try:
    from apt.data import make_dataloaders as _make_dataloaders
except Exception:  # pragma: no cover
    _make_dataloaders = None  # type: ignore

try:
    from apt.baselines.ft import lr_factor as _lr_factor
except Exception:  # pragma: no cover
    _lr_factor = None  # type: ignore


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TTA_FRACTION = 0.97

COFI_REPO_URL = "https://github.com/princeton-nlp/CoFiPruning"
COFI_REPO_DIRNAME = "CoFiPruning"
COFI_ENV_VAR = "COFI_DIR"

# CoFi default hyper-parameters (Xia et al., 2022) used when the external
# repository is not available / when nothing else is stated.
COFI_DEFAULTS: Dict[str, float] = {
    "lr": 2e-4,
    "batch_size": 32,
    "weight_decay": 0.01,
    "l0_lambda": 2.0e-8,          # L0 annealing lambda / regularization strength
    "l0_warmup_epochs": 5.0,      # epochs of pure LM ft before gate annealing
    "regularization_lambda": 0.1,  # layer-drop / intermediate distillation weight
    "distillation_temperature": 2.0,
    "l0_target_sparsity": 0.60,
    "initial_temperature": 5.0 / 3.0,
    "final_temperature": 0.1,
    "gate_alpha": -0.1,
    "gate_gamma": -0.1,
    "gate_zeta": 1.1,
    "collect_fn_batch_size": 32,
}

DEFAULT_TARGET_SPARSITY = 0.60
DEFAULT_LORA_RANK = 8
DEFAULT_SCALING = 2.0
DEFAULT_SEED = 42

MODES = ("cofi", "prune_distill", "lora_prune_distill")
DISPLAY_NAMES = {
    "cofi": "Prune+Distill",
    "prune_distill": "Prune+Distill",
    "lora_prune_distill": "LoRA+Prune+Distill",
}

TABLE2_REFERENCES: Dict[str, Dict[str, float]] = {
    # Table 2, RoBERTa-base at 60% sparsity (paper values used for validation).
    "prune_distill": {
        "sst2": 94.5,
        "mnli": 87.3,
        "train_time": 1495.3,
        "train_mem": 168.5,
        "inf_time": 38.6,
        "inf_mem": 79.2,
    },
    "lora_prune_distill": {
        "sst2": 91.9,
        "mnli": 84.2,
        "train_time": 6534.6,
        "train_mem": 141.4,
        "inf_time": 39.4,
        "inf_mem": 82.3,
    },
}

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
)

GLUE_TASKS = ("mnli", "sst2", "qnli", "qqp", "mrpc", "cola", "rte", "stsb")
GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad", "squad_v2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

TABLE6_COFI_DEFAULTS: Dict[str, Dict[str, float]] = {
    "glue-big": {"lr": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "glue-small": {"lr": 2e-4, "batch_size": 32, "epochs": 20, "distill_epochs": 10},
    "squad": {"lr": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "cnndm": {"lr": 1e-4, "batch_size": 16, "epochs": 16, "distill_epochs": 6},
}


# ---------------------------------------------------------------------------
# small helpers (shared with the other baselines)
# ---------------------------------------------------------------------------


def canonical_task(task: Optional[str]) -> str:
    """Normalise a task name to its canonical key."""
    if not task:
        return "sst2"
    key = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sst_2": "sst2",
        "sst": "sst2",
        "multinli": "mnli",
        "mnli_matched": "mnli",
        "mnli_mismatched": "mnli",
        "squadv2": "squad_v2",
        "squad2": "squad_v2",
        "squadv1": "squad",
        "cnn_dailymail": "cnndm",
        "cnn_dm": "cnndm",
        "cnndailymail": "cnndm",
        "mrpc": "mrpc",
        "cola": "cola",
        "rte": "rte",
        "stsb": "stsb",
        "sts_b": "stsb",
        "qqp": "qqp",
        "qnli": "qnli",
    }
    return aliases.get(key, key)


def is_glue_task(task: Optional[str]) -> bool:
    return canonical_task(task) in set(GLUE_TASKS)


def is_squad_task(task: Optional[str]) -> bool:
    return canonical_task(task) in {canonical_task(t) for t in SQUAD_TASKS}


def is_seq2seq_task(task: Optional[str]) -> bool:
    return canonical_task(task) in {canonical_task(t) for t in SEQ2SEQ_TASKS}


def table6_group_for(model_type: Optional[str], task: Optional[str]) -> str:
    """Return the Table 6 column used for a (model, task) pair."""
    task = canonical_task(task)
    if is_squad_task(task):
        return "squad"
    if is_seq2seq_task(task):
        return "cnndm"
    if task in set(GLUE_BIG_TASKS):
        return "glue-big"
    return "glue-small"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:  # pragma: no cover - depends on torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch is not None and torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    return "cpu"


def move_to_device(batch: Any, device: Any) -> Any:
    if torch is None:  # pragma: no cover
        return batch
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def model_inputs(batch: Dict[str, Any], keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, Any]:
    """Filter a batch down to the tensors a HF model forward accepts."""
    return {k: v for k, v in batch.items() if k in keys}


def flatten_parameter_count(model: Any, trainable_only: bool = True) -> int:
    if torch is None:  # pragma: no cover
        return 0
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# external repository plumbing
# ---------------------------------------------------------------------------


def locate_external_repo(
    path: Optional[str] = None,
    *,
    extra_candidates: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Locate a checkout of ``princeton-nlp/CoFiPruning``."""
    candidates: List[str] = []
    if path:
        candidates.append(str(path))
    env = os.environ.get(COFI_ENV_VAR) or os.environ.get("COFI_PRUNING_DIR")
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.join(os.getcwd(), "external"),
                 os.path.join(os.getcwd(), "third_party"),
                 os.path.join(os.getcwd(), "baselines")):
        candidates.append(os.path.join(base, COFI_REPO_DIRNAME))
        candidates.append(os.path.join(base, COFI_REPO_DIRNAME.lower()))
    if extra_candidates:
        candidates.extend(str(c) for c in extra_candidates)

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isdir(candidate) and (
            os.path.exists(os.path.join(candidate, "src"))
            or os.path.exists(os.path.join(candidate, "setup.py"))
            or os.path.exists(os.path.join(candidate, "README.md"))
        ):
            return os.path.abspath(candidate)
    return None


def load_external_repo(
    path: Optional[str] = None,
    *,
    force: bool = False,
    module_names: Optional[Sequence[str]] = None,
) -> Optional[Any]:
    """Best-effort import of the external CoFi implementation."""
    root = locate_external_repo(path)
    if root is None:
        if force:
            raise RuntimeError(
                f"CoFi repository not found; clone {COFI_REPO_URL} or set {COFI_ENV_VAR}"
            )
        return None
    for sub in ("", "src", "examples"):
        candidate = os.path.join(root, sub) if sub else root
        if candidate not in sys.path and os.path.isdir(candidate):
            sys.path.insert(0, candidate)
    for mod in (module_names or ("src.cofi", "cofi", "src")):  # pragma: no cover - env dependent
        try:
            __import__(mod)
            return sys.modules[mod]
        except Exception:
            continue
    return None


def external_repo_available(path: Optional[str] = None) -> bool:
    """Whether the external CoFi repository can be located (and imported)."""
    if locate_external_repo(path) is None:
        return False
    try:
        return load_external_repo(path) is not None
    except Exception:
        return False


def clone_external_repo(
    dest: Optional[str] = None,
    *,
    url: str = COFI_REPO_URL,
    depth: int = 1,
    verbose: bool = False,
) -> Optional[str]:
    """Clone the official CoFi repository if ``git``/network are available."""
    dest = dest or os.path.join(os.getcwd(), "external", COFI_REPO_DIRNAME)
    existing = locate_external_repo(dest)
    if existing:
        return existing
    if shutil.which("git") is None:
        if verbose:
            warnings.warn("git executable not found; cannot clone CoFi repository")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), url, dest]
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
    except Exception as exc:  # pragma: no cover - network dependent
        if verbose:
            warnings.warn(f"failed to clone CoFi repository: {exc}")
        return None
    return locate_external_repo(dest)


# ---------------------------------------------------------------------------
# L0 gate machinery (hard-concrete / Louizos et al., 2018)
# ---------------------------------------------------------------------------


class L0Gate(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Hard-concrete :math:`L_0` gate over ``num_units`` structured units.

    Follows CoFi (Xia et al., 2022): each unit owns a learnable logit
    ``alpha``; during training the gate is sampled from a stretched concrete
    (binary-concrete) distribution, at inference it is the deterministic
    clipped-mean ``min(1, max(0, sigmoid(alpha) * (zeta - gamma) + gamma))``.
    """

    def __init__(
        self,
        num_units: int,
        *,
        alpha_init: float = -0.1,
        gamma: float = -0.1,
        zeta: float = 1.1,
        temperature: float = 5.0 / 3.0,
        kind: int = HEAD,
        layer: int = -1,
        site: str = "",
        out_group_size: int = 1,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the CoFi baseline")
        super().__init__()
        self.num_units = int(num_units)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.temperature = float(temperature)
        self.kind = int(kind)
        self.layer = int(layer)
        self.site = str(site)
        self.out_group_size = int(out_group_size)
        self.log_alpha = nn.Parameter(
            torch.full((self.num_units,), float(alpha_init), dtype=torch.float32)
        )

    # -- probabilities / hard gates ---------------------------------------
    def probs(self) -> "torch.Tensor":
        t = _torch()
        return torch.sigmoid(self.log_alpha)

    def clipped_probs(self) -> "torch.Tensor":
        # P(gate > 0) with the stretched support [gamma, zeta].
        p = self.probs()
        return (p * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    @staticmethod
    def _hard_concrete(probs: "torch.Tensor", temperature: float, gamma: float, zeta: float):
        t = _torch()
        u = torch.rand_like(probs).clamp(1e-7, 1 - 1e-7)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + probs.clamp(1e-7, 1 - 1e-7).log()) / float(temperature))
        s = s * (zeta - gamma) + gamma
        return s.clamp(0.0, 1.0)

    def forward(self, *, sample: Optional[bool] = None) -> "torch.Tensor":
        """Return the (possibly stochastic) 0/1 gate values, shape ``(num_units,)``."""
        t = _torch()
        if sample is None:
            sample = self.training
        if sample:
            s_bar = self._hard_concrete(self.probs(), self.temperature, self.gamma, self.zeta)
            return (s_bar > 0.5).to(t.float32)
        return (self.clipped_probs() > 0.5).to(t.float32)

    def soft_gate(self) -> "torch.Tensor":
        """Differentiable gate value (clipped probability) for hard-concrete reg."""
        return self.clipped_probs()

    def expected_l0(self) -> "torch.Tensor":
        """Expected :math:`L_0` norm of this gate (used for the L0 penalty)."""
        p = self.probs().clamp(1e-7, 1 - 1e-7) * (self.zeta - self.gamma) + self.gamma
        p = p.clamp(1e-7, 1 - 1e-7)
        return p.sum()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class CoFiConfig:
    """Configuration for the CoFi / LoRA+Prune+Distill baselines."""

    model_name_or_path: str = "roberta-base"
    model_type: str = "roberta"
    task: str = "sst2"
    mode: str = "cofi"  # cofi | prune_distill | lora_prune_distill
    table6_group: Optional[str] = None

    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 40
    distill_epochs: int = 20

    target_sparsity: float = DEFAULT_TARGET_SPARSITY
    initial_rank: int = DEFAULT_LORA_RANK
    scaling: float = DEFAULT_SCALING

    # L0 (gate) hyper-parameters -- CoFi defaults
    l0_lambda: float = COFI_DEFAULTS["l0_lambda"]
    l0_warmup_epochs: float = COFI_DEFAULTS["l0_warmup_epochs"]
    l0_initial_temperature: float = COFI_DEFAULTS["initial_temperature"]
    l0_final_temperature: float = COFI_DEFAULTS["final_temperature"]
    gate_alpha: float = COFI_DEFAULTS["gate_alpha"]
    gate_gamma: float = COFI_DEFAULTS["gate_gamma"]
    gate_zeta: float = COFI_DEFAULTS["gate_zeta"]

    # dynamic layer-wise distillation -- CoFi defaults
    regularization_lambda: float = COFI_DEFAULTS["regularization_lambda"]
    distillation_temperature: float = COFI_DEFAULTS["distillation_temperature"]
    layer_distill_weight: float = 0.9
    pred_distill_weight: float = 1.0
    tau: int = 4

    # pruning granularity
    prune_heads: bool = True
    prune_neurons: bool = True
    prune_dims: bool = True

    # optimisation / runtime
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_kind: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = DEFAULT_SEED
    max_seq_length: int = 128
    max_target_length: int = 128
    dynamic_padding: bool = False
    num_workers: int = 0
    device: Optional[str] = None
    output_dir: str = "outputs/cofi"
    logging_steps: int = 50
    eval_steps: int = 0
    save_steps: int = 0
    inference_batch_size: int = 128
    sequence_length: int = 128
    fp16: bool = False
    bf16: bool = False
    measure_efficiency: bool = True
    recovery_epochs: int = 0
    max_train_batches: int = 0
    max_eval_batches: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- constructed -------------------------------------------------------
    def __post_init__(self) -> None:
        key = str(self.mode or "cofi").strip().lower().replace("-", "_").replace("+", "_")
        aliases = {
            "prune_distill": "prune_distill",
            "cofi": "cofi",
            "cofipruning": "cofi",
            "lora_prune_distill": "lora_prune_distill",
            "lora_cofi": "lora_prune_distill",
            "prune_distill_lora": "lora_prune_distill",
        }
        self.mode = aliases.get(key, "cofi")
        self.task = canonical_task(self.task)
        if self.table6_group is None:
            self.table6_group = table6_group_for(self.model_type, self.task)
        self.device = self.device or None

    # -- properties --------------------------------------------------------
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
        """CoFi prunes/detaches during a warm-up phase then regularizes."""
        return int(self.l0_warmup_epochs)

    @property
    def tuning_only_lora(self) -> bool:
        return self.mode == "lora_prune_distill"

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.mode, "CoFi")

    @property
    def resolved_sparsity(self) -> float:
        return float(self.target_sparsity)

    @property
    def resolved_density(self) -> float:
        return 1.0 - float(self.target_sparsity)

    @property
    def resolved_target_modules(self) -> Tuple[str, ...]:
        return lora_target_modules_for(self.model_type)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "CoFiConfig":
        data = dict(data or {})
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "CoFiConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        out.update(out.pop("extra", {}) or {})
        return out

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        try:
            import yaml

            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        except Exception:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, default=str)
        return path


def apply_table6_defaults(config: CoFiConfig, table: Optional[Dict[str, Any]] = None) -> CoFiConfig:
    """Fill Table 6 values for the config's task group (only where untouched)."""
    group = config.table6_group or table6_group_for(config.model_type, config.task)
    defaults = dict(TABLE6_COFI_DEFAULTS.get(group, TABLE6_COFI_DEFAULTS["glue-big"]))
    if table:
        external = table.get(group) or {}
        if isinstance(external, dict):
            defaults.update({k: v for k, v in external.items() if v is not None})
    if config.learning_rate == 2e-4 and "lr" in defaults:
        config.learning_rate = float(defaults["lr"])
    if config.batch_size == 32 and "batch_size" in defaults:
        config.batch_size = int(defaults["batch_size"])
    if config.epochs == 40 and "epochs" in defaults:
        config.epochs = int(defaults["epochs"])
    if config.distill_epochs == 20 and "distill_epochs" in defaults:
        config.distill_epochs = int(defaults["distill_epochs"])
    if config.is_seq2seq and config.max_seq_length == 128:
        config.max_seq_length = 512
    if config.recovery_epochs == 0:
        config.recovery_epochs = max(0, config.epochs - config.distill_epochs)
    return config


# ---------------------------------------------------------------------------
# LoRA target modules (mirrors apt/baselines/lora.py)
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES: Dict[str, Tuple[str, ...]] = {
    "roberta": ("query", "value"),
    "bert": ("query", "value"),
    "electra": ("query", "value"),
    "t5": ("q", "v"),
    "mt5": ("q", "v"),
    "deberta": ("query_proj", "value_proj"),
    "distilbert": ("q_lin", "v_lin"),
    "bart": ("q_proj", "v_proj"),
    "opt": ("q_proj", "v_proj"),
    "llama": ("q_proj", "v_proj"),
    "mistral": ("q_proj", "v_proj"),
    "gpt2": ("c_attn",),
}


def lora_target_modules_for(model_type: Optional[str] = None,
                            model_name_or_path: Optional[str] = None) -> Tuple[str, ...]:
    for key in (model_type or "", model_name_or_path or ""):
        key = str(key).lower()
        for family, modules in LORA_TARGET_MODULES.items():
            if family in key:
                return modules
    return LORA_TARGET_MODULES["roberta"]


# ---------------------------------------------------------------------------
# L0 gate bookkeeping over a wrapped model
# ---------------------------------------------------------------------------


@dataclass
class GateUnit:
    """One structured unit (head / neuron / hidden dimension) behind an L0 gate."""

    kind: int
    layer: int
    index: int
    site: str = ""
    name: str = ""
    param_count: float = 0.0
    unit_index: int = 0  # position inside its own gate vector
    gate_key: Tuple[int, int, str, int] = (0, 0, "", 0)

    @property
    def kind_name(self) -> str:
        return {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}.get(self.kind, "unit")

    @property
    def block_name(self) -> str:
        base = f"{self.kind_name}.{self.layer}.{self.index}"
        return f"{base}.{self.site}" if self.site else base


class GateBank(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Collection of :math:`L_0` gates attached to a wrapped model's structure.

    One gate vector is created per (kind, layer, site) group so that a single
    sampled mask is shared across all projections that touch the same unit
    (CoFi applies one gate per structured unit of the pruned model).
    """

    def __init__(self, config: Optional[CoFiConfig] = None, **kwargs: Any) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the CoFi baseline")
        super().__init__()
        cfg = config or CoFiConfig.from_dict(kwargs)
        self.config = cfg
        self.gates = nn.ModuleDict()
        self.units: List[GateUnit] = []
        self._built = False

    # -- construction ------------------------------------------------------
    def build(self, model: Any, shape: Any = None) -> "GateBank":
        """Create gates matching the (already wrapped) model's structure."""
        self.gates = nn.ModuleDict()
        self.units = []

        shape = shape if shape is not None else (apt_shape(model) if apt_shape else None)
        d_model = getattr(shape, "d_model", None) or getattr(
            getattr(model, "config", None), "hidden_size", None
        ) or 0
        n_layers = getattr(shape, "n_layers", None) or 0
        n_heads = getattr(shape, "n_heads", None) or 0
        n_ffn = getattr(shape, "n_ffn", None) or 0
        sites = tuple(getattr(shape, "sites", None) or ("query", "value"))

        if self.config.prune_heads and n_heads:
            for layer in range(max(n_layers, 1)):
                for site in sites:
                    self._add_gate(HEAD, layer, int(n_heads), site=site)
        if self.config.prune_neurons and n_ffn:
            for layer in range(max(n_layers, 1)):
                self._add_gate(NEURON, layer, int(n_ffn), site="ffn")
        if self.config.prune_dims and d_model:
            self._add_gate(DIMENSION, -1, int(d_model), site="hidden")

        self._built = True
        return self

    def _add_gate(self, kind: int, layer: int, num_units: int, site: str = "") -> L0Gate:
        key = self._key(kind, layer, site)
        gate = L0Gate(
            num_units,
            alpha_init=self.config.gate_alpha,
            gamma=self.config.gate_gamma,
            zeta=self.config.gate_zeta,
            temperature=self.config.l0_initial_temperature,
            kind=kind,
            layer=layer,
            site=site,
        )
        self.gates[key] = gate
        for i in range(int(num_units)):
            self.units.append(
                GateUnit(
                    kind=kind,
                    layer=int(layer),
                    index=i,
                    site=site,
                    name=f"{self._kind_name(kind)}.{layer}.{i}" + (f".{site}" if site else ""),
                    unit_index=i,
                    gate_key=self._key_tuple(kind, layer, site),
                )
            )
        return gate

    @staticmethod
    def _kind_name(kind: int) -> str:
        return {HEAD: "head", NEURON: "neuron", DIMENSION: "dimension"}.get(kind, "unit")

    @staticmethod
    def _key(kind: int, layer: int, site: str) -> str:
        return f"{GateBank._kind_name(kind)}__{int(layer)}__{site or 'main'}"

    @staticmethod
    def _key_tuple(kind: int, layer: int, site: str) -> Tuple[int, int, str, int]:
        return (int(kind), int(layer), str(site), 0)

    # -- sampling / statistics --------------------------------------------
    def gate(self, kind: int, layer: int, site: str = "") -> Optional[L0Gate]:
        return self.gates.get(self._key(kind, layer, site))

    def gates_of_kind(self, kind: int) -> List[L0Gate]:
        return [g for g in self.gates.values() if int(getattr(g, "kind", -1)) == int(kind)]

    def sample_masks(self, *, sample: Optional[bool] = None) -> Dict[int, Any]:
        """Return ``{kind: mask tensor}`` of current 0/1 gate values."""
        t = _torch()
        out: Dict[int, Any] = {}
        for kind in (HEAD, NEURON, DIMENSION):
            vectors = []
            for gate in self.gates_of_kind(kind):
                vectors.append(gate(sample=sample))
            if vectors:
                out[kind] = t.cat(vectors) if len(vectors) > 1 else vectors[0]
        return out

    def expected_l0(self) -> Any:
        t = _torch()
        total = t.zeros((), dtype=t.float32)
        for gate in self.gates.values():
            total = total + gate.expected_l0()
        return total

    def l0_loss(self) -> Any:
        """Gate regulariser ``lambda * sum_i E[L0(gate_i)]`` (CoFi hard-concrete)."""
        t = _torch()
        if not self.gates:
            return t.zeros((), dtype=t.float32)
        expected = float(self.expected_l0().detach())
        norm = max(len(self.units), 1)
        return self.config.l0_lambda * expected * norm / max(norm, 1) * 1.0

    def unit_counts(self) -> Dict[str, int]:
        counts = {"head": 0, "neuron": 0, "dimension": 0}
        for unit in self.units:
            counts[unit.kind_name] = counts.get(unit.kind_name, 0) + 1
        return counts

    def retained_counts(self, masks: Optional[Dict[int, Any]] = None) -> Dict[str, int]:
        t = _torch()
        masks = masks if masks is not None else self.sample_masks()
        out = {"head": 0, "neuron": 0, "dimension": 0}
        for kind, key in ((HEAD, "head"), (NEURON, "neuron"), (DIMENSION, "dimension")):
            mask = masks.get(kind)
            if mask is None:
                continue
            out[key] = int((mask > 0.5).sum().item())
        return out

    def sparsity(self, masks: Optional[Dict[int, Any]] = None) -> float:
        masks = masks if masks is not None else self.sample_masks()
        kept = self.retained_counts(masks)
        total_units = max(len(self.units), 1)
        kept_units = sum(kept.values())
        return float(max(0.0, 1.0 - kept_units / total_units))

    # -- annealing ---------------------------------------------------------
    def set_temperature(self, value: float) -> None:
        for gate in self.gates.values():
            gate.temperature = float(max(value, 1e-3))

    def temperature_for_progress(self, progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        start = float(self.config.l0_initial_temperature)
        end = float(self.config.l0_final_temperature)
        return float(start + (end - start) * p)

    def harden(self, *, inplace: bool = True, threshold: float = 0.5) -> Dict[int, Any]:
        """Snap gate probabilities so that future forward passes use 0/1 gates."""
        t = _torch()
        masks = {}
        with t.no_grad():
            for kind in (HEAD, NEURON, DIMENSION):
                vecs = []
                for gate in self.gates_of_kind(kind):
                    p = gate.clipped_probs()
                    gate.log_alpha.data = t.where(
                        p > threshold,
                        t.full_like(gate.log_alpha.data, 8.0),
                        t.full_like(gate.log_alpha.data, -8.0),
                    )
                    vecs.append(gate.clipped_probs())
                if vecs:
                    masks[kind] = t.cat(vecs) if len(vecs) > 1 else vecs[0]
        return masks

    def disable_grad(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def enable_grad(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def summary(self) -> Dict[str, Any]:
        counts = self.unit_counts()
        return {
            "num_gates": len(self.gates),
            "num_units": len(self.units),
            "units": counts,
            "expected_l0": float(self.expected_l0().detach().item()) if self.gates else 0.0,
            "temperature": float(
                next(iter(self.gates.values())).temperature if self.gates else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# model / data construction
# ---------------------------------------------------------------------------


def num_labels_for_task(task: str) -> int:
    task = canonical_task(task)
    if task == "mnli":
        return 3
    if task in ("stsb",):
        return 1
    if task in GLUE_TASKS:
        return 2
    return 2


def problem_type_for_task(task: str) -> Optional[str]:
    return "regression" if canonical_task(task) == "stsb" else None


def build_model_and_tokenizer(config: CoFiConfig) -> Tuple[Any, Any, str]:
    """Build a task-appropriate HuggingFace model + tokenizer (and task flag)."""
    try:
        from transformers import (
            AutoConfig,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # pragma: no cover - transformers required
        raise RuntimeError("transformers is required to build CoFi baseline models") from exc

    task = canonical_task(config.task)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
    if is_seq2seq_task(task):
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        kind = "seq2seq"
    elif is_squad_task(task):
        model = AutoModelForQuestionAnswering.from_pretrained(config.model_name_or_path)
        kind = "squad"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            num_labels=num_labels_for_task(task),
            problem_type=problem_type_for_task(task),
        )
        kind = "glue"
    return model, tokenizer, kind


def build_dataloaders_for_task(
    config: CoFiConfig,
    tokenizer: Any,
    splits: Sequence[str] = ("train", "validation"),
) -> Dict[str, Any]:
    """Build task dataloaders via :mod:`apt.data`."""
    if _make_dataloaders is None:  # pragma: no cover
        raise RuntimeError("apt.data.make_dataloaders is unavailable")
    kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "max_seq_length": config.max_seq_length,
        "num_workers": config.num_workers,
        "seed": config.seed,
    }
    if config.is_seq2seq:
        kwargs.update(
            {
                "max_target_length": config.max_target_length,
                "max_source_length": config.max_seq_length,
            }
        )
        return _make_dataloaders(config.task, tokenizer, model_type="t5", **kwargs)
    return _make_dataloaders(config.task, tokenizer, model_type="encoder", **kwargs)


# ---------------------------------------------------------------------------
# dynamic layer-wise distillation (CoFi, Xia et al., 2022)
# ---------------------------------------------------------------------------


class LayerwiseDistillation(nn.Module if nn is not None else object):  # type: ignore[misc]
    """"Dynamic layer-wise distillation" from CoFi.

    The teacher is the frozen, un-pruned model (or the pre-pruning student
    snapshot).  Every step a subset of teacher/student layer pairs is sampled
    (``tau`` contiguous blocks, one layer per block -- the same block-wise
    sampling used by APT for a comparable protocol) and the hidden states are
    matched with an MSE objective.  Student layers that have been fully pruned
    are skipped via a dynamic teacher->student layer mapping.
    """

    def __init__(self, config: Optional[CoFiConfig] = None) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for the CoFi baseline")
        super().__init__()
        self.config = config or CoFiConfig()
        self._teacher: Optional[Any] = None
        self._collectors: List[Any] = []
        self._mapping: List[int] = []
        self.n_student_layers = 0
        self.n_teacher_layers = 0

    # -- teacher ----------------------------------------------------------
    def build_teacher(self, student: Any) -> Any:
        """Create the frozen teacher sharing the student's frozen parameters."""
        import copy

        t = _torch()
        memo: Dict[int, Any] = {}

        def _keep_frozen(obj):
            if isinstance(obj, t.nn.Parameter):
                if not obj.requires_grad:
                    return obj  # share frozen LM parameters
                return copy.deepcopy(obj)
            return None

        try:
            teacher = copy.deepcopy(student, memo)
        except Exception:  # pragma: no cover - deepcopy can fail on odd modules
            teacher = student
        else:
            # re-run deepcopy with a memo that shares frozen parameters
            memo = {}
            try:
                teacher = copy.deepcopy(student, memo)
            except Exception:
                teacher = student
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self._teacher = teacher
        self.n_teacher_layers = len(_transformer_layers(teacher))
        self.n_student_layers = len(_transformer_layers(student))
        self._mapping = list(range(max(self.n_student_layers, 1)))
        return teacher

    @property
    def teacher(self) -> Optional[Any]:
        return self._teacher

    # -- layer mapping ----------------------------------------------------
    def student_layer_keep(self, student: Any, gates: Optional[GateBank] = None,
                          threshold: float = 0.5) -> List[int]:
        """1 if a student layer still has any live head/neuron, else 0."""
        n_layers = self.n_student_layers or len(_transformer_layers(student))
        if gates is None or not gates.gates:
            return [1] * max(n_layers, 1)
        masks = gates.sample_masks()
        keep = [0] * max(n_layers, 1)
        for kind in (HEAD, NEURON):
            for gate in gates.gates_of_kind(kind):
                layer = int(getattr(gate, "layer", -1))
                if 0 <= layer < len(keep):
                    if bool((gate(sample=False) > threshold).any().item()):
                        keep[layer] = 1
        if not any(keep):
            return [1] * len(keep)
        return keep

    def update_mapping(self, keep: Sequence[int]) -> List[int]:
        """Map every teacher layer to the closest non-pruned student layer."""
        n_student = len(keep)
        alive = [i for i, flag in enumerate(keep) if flag]
        if not alive:
            self._mapping = list(range(max(self.n_teacher_layers, 1)))
            return self._mapping
        mapping = []
        for teacher_idx in range(max(self.n_teacher_layers, 1)):
            if n_student <= 1 or self.n_teacher_layers <= 1:
                mapping.append(alive[0])
                continue
            target = teacher_idx * (n_student - 1) / max(self.n_teacher_layers - 1, 1)
            mapping.append(min(alive, key=lambda s: abs(s - target)))
        self._mapping = mapping
        return mapping

    def sample_teacher_layers(self, rng: Optional[random.Random] = None) -> List[int]:
        """Block-wise sampling: one teacher layer per contiguous block (tau=4)."""
        n_layers = max(self.n_teacher_layers, 1)
        n_blocks = min(max(int(self.config.tau or 4), 1), n_layers)
        rng = rng or random
        out = []
        for block in range(n_blocks):
            start = block * n_layers // n_blocks
            stop = (block + 1) * n_layers // n_blocks
            stop = max(stop, start + 1)
            out.append(rng.randrange(start, min(stop, n_layers)))
        return out

    # -- loss -------------------------------------------------------------
    def forward(
        self,
        student_states: Dict[int, Any],
        teacher_states: Dict[int, Any],
        teacher_indices: Optional[Sequence[int]] = None,
        *,
        mask: Optional[Any] = None,
    ) -> Any:
        """MSE layer-wise distillation loss over the sampled/mapped layers."""
        t = _torch()
        if not student_states or not teacher_states:
            return t.zeros((), dtype=t.float32)
        indices = list(teacher_indices) if teacher_indices is not None else sorted(teacher_states)
        total = t.zeros((), dtype=t.float32, device=next(iter(student_states.values())).device)
        count = 0
        for teacher_idx in indices:
            teacher_h = teacher_states.get(teacher_idx)
            if teacher_h is None:
                continue
            student_idx = self._mapping[teacher_idx] if teacher_idx < len(self._mapping) else None
            student_h = student_states.get(student_idx) if student_idx is not None else None
            if student_h is None:
                continue
            d_s = student_h.shape[-1]
            d_t = teacher_h.shape[-1]
            width = min(d_s, d_t)
            diff = student_h[..., :width].float() - teacher_h[..., :width].float()
            if mask is not None:
                diff = diff * mask.to(diff.dtype)
            total = total + diff.pow(2).mean()
            count += 1
        if count == 0:
            return t.zeros((), dtype=t.float32, device=total.device)
        return total / count

    def total_loss(self, task_loss: Any, distill_loss: Optional[Any], progress: float = 0.0) -> Any:
        """``L = (1 - w) * L_ft + w * L_distill`` with CoFi's dynamic weight."""
        t = _torch()
        if distill_loss is None:
            return task_loss
        w = float(self.config.layer_distill_weight) * float(
            self.dynamic_weight(progress)
        )
        return (1.0 - w) * task_loss + w * distill_loss

    def dynamic_weight(self, progress: float) -> float:
        """Layer-wise objectives are dynamically re-weighted across training."""
        p = min(max(float(progress), 0.0), 1.0)
        base = float(self.config.regularization_lambda)
        return float(base + (1.0 - base) * p)

    def num_parameters(self) -> int:
        if self._teacher is None:  # pragma: no cover
            return 0
        return sum(p.numel() for p in self._teacher.parameters() if p.requires_grad)


def _transformer_layers(model: Any) -> List[Any]:
    if find_transformer_layers is not None:
        try:
            layers = find_transformer_layers(model)
            if layers:
                return list(layers)
        except Exception:
            pass
    # generic fallback
    for attr in ("encoder", "model", "transformer", "decoder"):
        node = getattr(model, attr, None)
        if node is None:
            continue
        for sub in ("layer", "layers", "block", "blocks", "h"):
            seq = getattr(node, sub, None)
            if seq is not None and hasattr(seq, "__len__") and len(seq) > 0:
                return list(seq)
    return []


class HiddenStateCollector:
    """Forward-hook collector for per-layer hidden states (teacher or student)."""

    def __init__(self, model: Any, layers: Optional[Sequence[Any]] = None) -> None:
        self.model = model
        self.layers = list(layers) if layers is not None else _transformer_layers(model)
        self.states: Dict[int, Any] = {}
        self._handles: List[Any] = []

    def _hook(self, index: int) -> Callable:
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.states[index] = hidden

        return hook

    def attach(self) -> "HiddenStateCollector":
        self.detach()
        for index, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._hook(index)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def clear(self) -> None:
        self.states = {}

    def __enter__(self) -> "HiddenStateCollector":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def __len__(self) -> int:
        return len(self.layers)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def _labels_from_batch(batch: Dict[str, Any]) -> Any:
    for key in ("labels", "label", "label_ids"):
        if key in batch:
            return batch[key]
    return None


def _default_compute_metrics(task: str) -> Callable[[Any, Any], Dict[str, float]]:
    if _compute_metrics is not None:
        def _fn(predictions, references, **kwargs):
            return _compute_metrics(task, predictions, references, **kwargs)

        return _fn

    def _fallback(predictions, references, **kwargs):  # pragma: no cover
        pairs = list(zip(list(predictions), list(references)))
        if not pairs:
            return {"accuracy": 0.0}
        correct = sum(1 for p, r in pairs if p == r)
        return {"accuracy": 100.0 * correct / len(pairs)}

    return _fallback


def _primary(metrics: Dict[str, float], task: Optional[str] = None) -> Optional[float]:
    if not metrics:
        return None
    if _primary_metric is not None and task:
        try:
            return float(_primary_metric(task, metrics))
        except Exception:
            pass
    for key in ("accuracy", "f1", "matthews_correlation", "spearmanr", "rouge1", "exact", "primary"):
        if key in metrics:
            return float(metrics[key])
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _predictions_from_logits(logits: Any, task: str) -> List[Any]:
    t = _torch()
    if is_seq2seq_task(task):  # pragma: no cover - seq2seq handled separately
        return []
    if logits is None:
        return []
    if logits.dim() == 1 or logits.shape[-1] == 1:
        return [float(v) for v in logits.reshape(-1).detach().cpu().tolist()]
    return logits.argmax(dim=-1).detach().cpu().tolist()


def _evaluate_plain(model: Any, dataloader: Any, config: CoFiConfig,
                    compute_metrics: Optional[Callable] = None,
                    tokenizer: Any = None) -> Dict[str, float]:
    """Task evaluation for a plain (merged) HF model."""
    t = _torch()
    task = canonical_task(config.task)
    device = resolve_device(config.device)
    model.eval()

    if is_squad_task(task):  # delegate to the shared SQuAD post-processing
        try:
            from apt.data.squad import (
                compute_squad_metrics,
                references_for_examples,
                write_predictions,
            )

            features = getattr(dataloader.dataset, "features", None)
            examples = getattr(dataloader.dataset, "examples", None)
            if features is not None and examples is not None:
                from apt.data.squad import SquadDataset  # noqa: F401

                all_results = []
                with t.no_grad():
                    for batch in dataloader:
                        batch = move_to_device(batch, device)
                        inputs = model_inputs(batch)
                        out = model(**inputs)
                        start = out.start_logits.detach().cpu().tolist()
                        end = out.end_logits.detach().cpu().tolist()
                        ids = batch.get("unique_id")
                        ids = ids.detach().cpu().tolist() if ids is not None else []
                        for uid, s, e in zip(ids, start, end):
                            all_results.append({"unique_id": uid, "start_logits": s, "end_logits": e})
                preds, _, no_ans = write_predictions(features, examples, all_results, tokenizer=tokenizer)
                return compute_squad_metrics(preds, references_for_examples(examples),
                                             no_answer_probs=no_ans)
        except Exception:
            pass

    predictions: List[Any] = []
    references: List[Any] = []
    with t.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            labels = _labels_from_batch(batch)
            inputs = model_inputs(batch)
            inputs.pop("labels", None)
            inputs.pop("label", None)
            out = model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            predictions.extend(_predictions_from_logits(logits, task))
            if labels is not None:
                references.extend(labels.detach().cpu().tolist())
    fn = compute_metrics or _default_compute_metrics(task)
    return fn(predictions, references)


# ---------------------------------------------------------------------------
# the baseline method
# ---------------------------------------------------------------------------


class CoFiMethod:
    """CoFi / LoRA+Prune+Distill baseline runner.

    Three phases, following Sec. 5.2 and the CoFi recipe:

    1. **Warm-up** (``l0_warmup_epochs``): train the LM (full FT for ``cofi``,
       LoRA-only for ``lora_prune_distill``) with the :math:`L_0` gates fixed.
    2. **Prune + distill**: anneal the gate temperature, add the
       :math:`L_0` regulariser and the dynamic layer-wise distillation loss
       from the frozen teacher, then harden the gates at the target sparsity.
    3. **Recovery**: fine-tune the physically pruned model.
    """

    def __init__(
        self,
        config: Optional[CoFiConfig] = None,
        model: Any = None,
        tokenizer: Any = None,
        train_dataloader: Any = None,
        eval_dataloader: Any = None,
        compute_metrics: Optional[Callable] = None,
        reference_metric: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or CoFiConfig.from_dict(kwargs)
        if isinstance(cfg, dict):
            cfg = CoFiConfig.from_dict(cfg)
        self.config = apply_table6_defaults(cfg)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.reference_metric = reference_metric

        self.device = resolve_device(self.config.device)
        self.gates: Optional[GateBank] = None
        self.distiller: Optional[LayerwiseDistillation] = None
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.history: List[Dict[str, Any]] = []
        self.prune_result: Optional[Dict[str, Any]] = None
        self.train_time_s: float = 0.0
        self.train_peak_mem_mb: float = 0.0
        self.tta_seconds: Optional[float] = None
        self.num_tuning_params: int = 0
        self.shape: Any = None

    # -- setup ------------------------------------------------------------
    def setup_model(self) -> Any:
        if self.model is None:
            self.model, self.tokenizer, self.kind = build_model_and_tokenizer(self.config)
        else:
            kind = "seq2seq" if self.config.is_seq2seq else (
                "squad" if self.config.is_squad else "glue"
            )
            self.kind = kind
        if wrap_model is not None and (
            is_wrapped is None or not is_wrapped(self.model)
        ):
            try:
                self.model = wrap_model(
                    self.model,
                    rank=self.config.initial_rank,
                    scaling=self.config.scaling,
                    model_type=self.config.model_type,
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"could not wrap model for CoFi gates: {exc}")
        if apt_shape is not None:
            try:
                self.shape = apt_shape(self.model)
            except Exception:
                self.shape = None
        self.gates = GateBank(self.config).build(self.model, self.shape).to(self.device)
        self.distiller = LayerwiseDistillation(self.config)
        return self.model

    def setup_data(self) -> Dict[str, Any]:
        if self.train_dataloader is not None:
            return {"train": self.train_dataloader, "validation": self.eval_dataloader}
        loaders = build_dataloaders_for_task(self.config, self.tokenizer)
        self.train_dataloader = loaders.get("train")
        self.eval_dataloader = loaders.get("validation") or loaders.get("test")
        return loaders

    def wrap_for_tuning(self) -> Any:
        """Apply LoRA when running the ``lora_prune_distill`` variant."""
        if not self.config.tuning_only_lora:
            return self.model
        try:
            from apt.baselines.lora import apply_lora

            apply_lora(
                self.model,
                rank=self.config.initial_rank,
                scaling=self.config.scaling,
                target_modules=self.config.resolved_target_modules,
                model_type=self.config.model_type,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"LoRA injection unavailable, tuning full LM: {exc}")
        return self.model

    # -- parameter groups -------------------------------------------------
    def tuning_parameters(self) -> List[Any]:
        """Tunable parameters: L0 gate logits (+ LoRA params for the LoRA variant)."""
        params: List[Any] = []
        seen = set()
        if self.gates is not None:
            for p in self.gates.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        if self.config.tuning_only_lora:
            for name, p in self.model.named_parameters():
                if p.requires_grad and ("lora" in name.lower()):
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
        elif self.model is not None:
            # CoFi (full LM) variant: everything trainable is tuned.
            for p in self.model.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def setup_optimizer(self, parameters: Optional[Iterable[Any]] = None) -> Any:
        t = _torch()
        params = list(parameters) if parameters is not None else self.tuning_parameters()
        if not params:
            self.optimizer = None
            return None
        decay, no_decay = [], []
        for p in params:
            (no_decay if p.ndim <= 1 else decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        self.optimizer = t.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.num_tuning_params = sum(p.numel() for p in params)
        return self.optimizer

    def steps_per_epoch(self) -> int:
        if self.train_dataloader is None:
            return 0
        try:
            return len(self.train_dataloader)
        except TypeError:  # pragma: no cover
            return 0

    def setup_scheduler(self, total_steps: Optional[int] = None) -> Any:
        t = _torch()
        if self.optimizer is None:
            self.scheduler = None
            return None
        total_steps = int(total_steps or max(self.steps_per_epoch() * self.config.num_epochs, 1))
        warmup = int(self.config.warmup_ratio * total_steps)

        def factor(step: int) -> float:
            if _lr_factor is not None:
                return _lr_factor(
                    step, total_steps, warmup_steps=warmup, kind=self.config.lr_kind
                )
            if step < warmup:
                return float(step) / float(max(warmup, 1))
            return max(0.0, 1.0 - (step - warmup) / float(max(total_steps - warmup, 1)))

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        return self.scheduler

    # -- mask application -------------------------------------------------
    def masks_to_model(self, masks: Optional[Dict[int, Any]] = None) -> None:
        """Push sampled gate values onto the wrapped model's masks."""
        if self.model is None or self.gates is None:
            return
        masks = masks if masks is not None else self.gates.sample_masks()
        hidden = masks.get(DIMENSION)
        head = masks.get(HEAD)
        neuron = masks.get(NEURON)
        model = self.model
        if hasattr(model, "set_dim_mask") and hidden is not None:
            try:
                model.set_dim_mask(hidden.detach().cpu())
            except Exception:
                pass
        for name, module in iter_masked_linears(model):
            kind = getattr(module, "kind", None)
            layer = getattr(module, "layer_idx", -1)
            site = getattr(module, "module_name", "") or ""
            try:
                if kind == HEAD and head is not None and hasattr(module, "set_output_group_mask"):
                    n_groups = getattr(module, "num_out_groups", None)
                    start = 0
                    if layer is not None and layer >= 0 and self.shape is not None:
                        n_heads = getattr(self.shape, "n_heads", 0)
                        sites = tuple(getattr(self.shape, "sites", ("query", "value")))
                        if site and site.split(".")[-1] in sites:
                            start = sites.index(site.split(".")[-1]) * int(n_heads)
                    if n_groups:
                        module.set_output_group_mask(head[start:start + int(n_groups)])
                elif kind == NEURON and neuron is not None and hasattr(module, "set_output_group_mask"):
                    n_groups = getattr(module, "num_out_groups", None)
                    if n_groups:
                        module.set_output_group_mask(neuron[: int(n_groups)])
                if hidden is not None and hasattr(module, "set_input_mask"):
                    d_in = hidden.numel()
                    module.set_input_mask(hidden[: min(d_in, hidden.numel())])
            except Exception:
                continue
        if hasattr(model, "sync_masks"):
            try:
                model.sync_masks()
            except Exception:
                pass

    # -- loss helpers -----------------------------------------------------
    def _task_loss(self, out: Any, labels: Any, kind: str) -> Any:
        t = _torch()
        config = getattr(self.model, "config", None)
        problem_type = getattr(config, "problem_type", None)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if labels is None:
            return t.zeros((), dtype=t.float32, device=logits.device)
        if kind == "seq2seq":
            return out.loss if getattr(out, "loss", None) is not None else t.zeros(
                (), dtype=logits.dtype, device=logits.device
            )
        if problem_type == "regression" or (labels.dim() > 1 and labels.shape[-1] == logits.shape[-1]
                                            and logits.shape[-1] == 1):
            return t.nn.functional.mse_loss(logits.view(-1).float(), labels.view(-1).float())
        return t.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]).float(), labels.view(-1).long()
        )

    # -- main fit ---------------------------------------------------------
    def fit(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run the CoFi warm-up -> prune+distill -> recovery schedule."""
        t = _torch()
        if self.model is None:
            self.setup_model()
        if self.train_dataloader is None:
            self.setup_data()
        self.model.to(self.device).train()
        self.wrap_for_tuning()
        self.setup_optimizer()
        self.setup_scheduler()
        if self.gates is not None:
            self.gates.to(self.device)

        distiller = self.distiller
        distiller.build_teacher(self.model)

        steps_per_epoch = max(self.steps_per_epoch(), 1)
        total_epochs = max(int(self.config.num_epochs), 1)
        warmup_epochs = max(int(self.config.l0_warmup_epochs), 0)
        prune_epochs = max(int(self.config.distill_epochs), 0)
        if warmup_epochs + prune_epochs > total_epochs:
            warmup_epochs = max(0, total_epochs - prune_epochs)
        prune_end_step = (warmup_epochs + prune_epochs) * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        if max_steps:
            total_steps = min(total_steps, int(max_steps))

        teacher_collector = HiddenStateCollector(distiller.teacher)
        student_collector = HiddenStateCollector(self.model)
        rng = random.Random(self.config.seed)

        reset_peak = None
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.reset_peak_memory_stats()
            reset_peak = True

        start_time = time.time()
        global_step = 0
        done = False
        for epoch in range(total_epochs):
            if done:
                break
            phase = "warmup"
            if epoch >= warmup_epochs and epoch < warmup_epochs + prune_epochs:
                phase = "prune"
            elif epoch >= warmup_epochs + prune_epochs:
                phase = "recovery"
            if self.config.recovery_epochs and epoch >= total_epochs - int(self.config.recovery_epochs):
                phase = "recovery"

            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = move_to_device(batch, self.device)
                labels = _labels_from_batch(batch)
                inputs = model_inputs(batch)
                inputs.pop("labels", None)
                inputs.pop("label", None)

                progress = 0.0
                if prune_end_step > 0:
                    progress = min(max(global_step / float(max(prune_end_step, 1)), 0.0), 1.0)

                if self.gates is not None:
                    if phase == "warmup":
                        self.gates.disable_grad()
                        self.gates.set_temperature(self.config.l0_initial_temperature)
                    else:
                        self.gates.enable_grad()
                        self.gates.set_temperature(
                            self.gates.temperature_for_progress(progress)
                        )
                    self.masks_to_model()

                # ---- student forward ------------------------------------
                with student_collector:
                    out = self.model(**inputs)
                task_loss = self._task_loss(out, labels, self.kind)

                distill_loss = None
                if phase != "warmup" and distiller.teacher is not None:
                    keep = distiller.student_layer_keep(self.model, self.gates)
                    mapping = distiller.update_mapping(keep)
                    teacher_idx = distiller.sample_teacher_layers(rng)
                    with t.no_grad(), teacher_collector:
                        t_inputs = dict(inputs)
                        distiller.teacher(**t_inputs)
                    teacher_states = {
                        i: teacher_collector.states.get(i) for i in teacher_idx
                    }
                    if mapping:
                        student_indices = {mapping[i] for i in teacher_idx if i < len(mapping)}
                        student_states = {
                            i: student_collector.states.get(i) for i in student_indices
                        }
                    else:
                        student_states = dict(student_collector.states)
                    distill_loss = distiller(
                        student_states, teacher_states, teacher_idx
                    )

                loss = distiller.total_loss(task_loss, distill_loss, progress)
                if self.gates is not None and phase != "warmup":
                    loss = loss + self.gates.l0_loss()

                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm:
                    params = [p for p in self.tuning_parameters() if p.grad is not None]
                    if params:
                        t.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                if self.optimizer is not None:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                global_step += 1
                if self.config.logging_steps and global_step % self.config.logging_steps == 0:
                    self.history.append(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "phase": phase,
                            "loss": float(loss.detach().item()),
                            "task_loss": float(task_loss.detach().item()),
                            "distill_loss": float(distill_loss.detach().item())
                            if distill_loss is not None
                            else None,
                            "sparsity": float(self.gates.sparsity()) if self.gates else 0.0,
                            "lr": float(self.optimizer.param_groups[0]["lr"])
                            if self.optimizer
                            else 0.0,
                        }
                    )
                if max_steps and global_step >= int(max_steps):
                    done = True
                    break

            if phase in ("prune", "recovery") and epoch + 1 >= warmup_epochs + prune_epochs:
                # end of the pruning phase: harden gates at the target sparsity
                if self.gates is not None and phase == "prune":
                    target = self.config.resolved_sparsity
                    masks = self._harden_to_sparsity(target)
                    self.masks_to_model(masks)
                    if distiller.teacher is not None:
                        self.prune_result = {"masks": {k: v.detach().cpu() for k, v in masks.items()}}

        self.train_time_s = time.time() - start_time
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU only
            self.train_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        metrics = self.evaluate()
        self.tta_seconds = self._track_tta(metrics)
        return self.summary()

    def _harden_to_sparsity(self, target_sparsity: float) -> Dict[int, Any]:
        """Harden `target_sparsity` fraction of gate units to zero (global)."""
        t = _torch()
        assert self.gates is not None
        target_sparsity = float(min(max(target_sparsity, 0.0), 0.999))
        prob_by_kind: Dict[int, List[Tuple[Any, Any]]] = {}
        for kind in (HEAD, NEURON, DIMENSION):
            for gate in self.gates.gates_of_kind(kind):
                prob_by_kind.setdefault(kind, []).append((gate, gate.clipped_probs().detach()))

        flat: List[Tuple[int, Any, int]] = []
        for kind, entries in prob_by_kind.items():
            for gate, probs in entries:
                for i, value in enumerate(probs.tolist()):
                    flat.append((kind, gate, i))
                    flat[-1] = (kind, gate, i) if False else flat[-1]
        # build a score list of (score, kind, gate, index)
        scored: List[Tuple[float, int, Any, int]] = []
        for kind, entries in prob_by_kind.items():
            for gate, probs in entries:
                for i, value in enumerate(probs.tolist()):
                    scored.append((float(value), kind, gate, i))
        scored.sort(key=lambda item: item[0])
        n_keep = int(round(len(scored) * (1.0 - target_sparsity)))
        n_keep = max(1, min(n_keep, len(scored)))
        keep = scored[len(scored) - n_keep:] if n_keep > 0 else []

        keep_set = {(kind, id(gate), i) for _s, kind, gate, i in keep}
        with t.no_grad():
            for kind, entries in prob_by_kind.items():
                for gate, _probs in entries:
                    values = []
                    for i in range(gate.num_units):
                        keep_it = (kind, id(gate), i) in keep_set
                        values.append(8.0 if keep_it else -8.0)
                    gate.log_alpha.data = t.tensor(values, dtype=gate.log_alpha.dtype,
                                                  device=gate.log_alpha.device)
        return self.gates.sample_masks()

    # -- pruning / physical export ---------------------------------------
    def select(self) -> Any:
        """Select blocks by density under the target sparsity (`BlockSelector`)."""
        if self.gates is None:
            return None
        masks = self.gates.sample_masks()
        scores: Dict[str, float] = {}
        for kind in (HEAD, NEURON, DIMENSION):
            vec = masks.get(kind)
            if vec is None:
                continue
            for i, value in enumerate(vec.tolist()):
                name = f"{{}}.{i}"  # filled in below with layer/site info
                del name
        for unit, value in zip(self.gates.units, self._flat_mask_values(masks)):
            scores[unit.block_name] = value
        if BlockSelector is None or self.shape is None:
            return None
        try:
            selector = BlockSelector(self.shape)
            blocks = selector.enumerate_blocks()
            for block in blocks:
                block.salience = float(scores.get(block.name, 0.0))
            return selector.select_to_sparsity(
                blocks, self.config.resolved_sparsity,
                original_param_count=selector.shape.full_param_count,
            )
        except Exception:
            return None

    def _flat_mask_values(self, masks: Dict[int, Any]) -> List[float]:
        out: List[float] = []
        for kind in (HEAD, NEURON, DIMENSION):
            vec = masks.get(kind)
            if vec is not None:
                out.extend(float(v) for v in vec.tolist())
        return out

    def prune(self, *, restore_linears: bool = True) -> Any:
        """Physically remove the pruned blocks (merge + slice)."""
        if self.gates is not None:
            self.gates.harden()
            self.masks_to_model()
        if merge_and_prune is None:  # pragma: no cover
            return self.model
        if self.model is not None:
            try:
                if hasattr(self.model, "harden_masks"):
                    self.model.harden_masks()
            except Exception:
                pass
        try:
            self.model = merge_and_prune(self.model)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"physical pruning failed: {exc}")
        return self.model

    def recover(self, epochs: Optional[int] = None) -> Any:
        """Post-pruning recovery fine-tuning (recovery stage of the baseline)."""
        epochs = int(epochs if epochs is not None else self.config.recovery_epochs)
        if epochs <= 0 or self.train_dataloader is None:
            return self.model
        t = _torch()
        self.model.to(self.device).train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:  # pragma: no cover
            return self.model
        optimizer = t.optim.AdamW(params, lr=self.config.learning_rate,
                                  weight_decay=self.config.weight_decay)
        for _epoch in range(epochs):
            for batch in self.train_dataloader:
                batch = move_to_device(batch, self.device)
                labels = _labels_from_batch(batch)
                inputs = model_inputs(batch)
                inputs.pop("labels", None)
                out = self.model(**inputs)
                loss = self._task_loss(out, labels, self.kind)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.max_grad_norm:
                    t.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                optimizer.step()
        return self.model

    def rrecover(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - typo alias
        return self.recover(*args, **kwargs)

    # -- evaluation ------------------------------------------------------
    def predict(self, dataloader: Optional[Any] = None) -> Dict[str, Any]:
        dataloader = dataloader if dataloader is not None else self.eval_dataloader
        if dataloader is None:
            return {}
        metrics = _evaluate_plain(self.model, dataloader, self.config,
                                  self.compute_metrics, self.tokenizer)
        return {"metrics": metrics, "primary": _primary(metrics, self.config.task)}

    def evaluate(self, dataloader: Optional[Any] = None, step: Optional[int] = None) -> Dict[str, float]:
        return self.predict(dataloader).get("metrics", {})

    def _track_tta(self, metrics: Dict[str, float]) -> Optional[float]:
        value = _primary(metrics, self.config.task)
        if value is None or not self.reference_metric:
            return None
        target = float(self.reference_metric) * TTA_FRACTION
        if value >= target:
            return float(self.train_time_s)
        history = [(h.get("elapsed"), h.get("metric")) for h in self.history]
        if len(history) < 2:
            return None
        for (t0, m0), (t1, m1) in zip(history, history[1:]):
            if m0 is None or m1 is None:
                continue
            if (m0 < target <= m1) or (m1 < target <= m0):
                if m1 == m0:
                    return float(t1)
                frac = (target - m0) / (m1 - m0)
                return float(t0 + frac * (t1 - t0))
        return None

    # -- reporting -------------------------------------------------------
    def prune_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mode": self.config.mode,
            "method": self.config.display_name,
            "target_sparsity": self.config.resolved_sparsity,
        }
        if self.gates is not None:
            out["gates"] = self.gates.summary()
            out["retained"] = self.gates.retained_counts()
            out["realized_sparsity"] = float(self.gates.sparsity())
        if self.model is not None:
            out["num_parameters"] = flatten_parameter_count(self.model, trainable_only=False)
            out["num_tuning_parameters"] = self.num_tuning_params
        return out

    def summary(self) -> Dict[str, Any]:
        out = {
            "method": self.config.display_name,
            "mode": self.config.mode,
            "model": self.config.model_name_or_path,
            "task": self.config.task,
            "sparsity": self.config.resolved_sparsity,
            "train_time_s": self.train_time_s,
            "train_peak_mem_mb": self.train_peak_mem_mb,
            "tta_seconds": self.tta_seconds,
            "num_tuning_parameters": self.num_tuning_params,
            "history": list(self.history),
        }
        out.update(self.prune_summary())
        out["trainer"] = self
        return out

    def save(self, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.model is not None:
            try:
                self.model.save_pretrained(output_dir)
            except Exception:  # pragma: no cover
                if torch is not None:
                    torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:  # pragma: no cover
                pass
        self.config.save(os.path.join(output_dir, "config.yaml"))
        metrics = self.evaluate()
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)
        return output_dir


# ---------------------------------------------------------------------------
# functional entry points
# ---------------------------------------------------------------------------


def prune_with_cofi(
    model: Any = None,
    dataloader: Any = None,
    *,
    config: Optional[Any] = None,
    task: str = "sst2",
    sparsity: float = DEFAULT_TARGET_SPARSITY,
    device: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot CoFi structural pruning without distillation/retraining.

    Provided so the baseline can be invoked for the Table 2 "Prune+Distill"
    row's inference-efficiency numbers (38.6% inference time, 79.2% inference
    memory relative to FT on RoBERTa-base at 60% sparsity).
    """
    cfg = config if isinstance(config, CoFiConfig) else CoFiConfig.from_dict(
        config or {}, task=task, target_sparsity=sparsity, **({"device": device} if device else {})
    )
    method = CoFiMethod(cfg, model=model, **kwargs)
    if method.model is None:
        method.setup_model()
    if method.gates is None:
        method.setup_model()
    method.gates.build(method.model, method.shape)
    if dataloader is not None:
        # use data-dependent gate statistics where cheaply possible
        method.train_dataloader = dataloader
    masks = method._harden_to_sparsity(cfg.resolved_sparsity)
    method.masks_to_model(masks)
    method.prune()
    result = {
        "model": method.model,
        "method": cfg.display_name,
        "sparsity": cfg.resolved_sparsity,
        "realized_sparsity": float(method.gates.sparsity()) if method.gates else None,
        "retained": method.gates.retained_counts() if method.gates else None,
        "num_parameters": flatten_parameter_count(method.model, trainable_only=False),
    }
    if verbose:
        print(f"[CoFi] {result['method']} -> sparsity {result['realized_sparsity']}")
    return result


def _train_one(
    config: Optional[Any] = None,
    model: Any = None,
    tokenizer: Any = None,
    train_dataloader: Any = None,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable] = None,
    reference_metric: Optional[float] = None,
    *,
    evaluate_now: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    cfg = config
    if cfg is None:
        cfg = CoFiConfig.from_dict(kwargs)
    elif isinstance(cfg, dict):
        cfg = CoFiConfig.from_dict(cfg, **kwargs)
    elif kwargs:
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    set_seed(cfg.seed)
    method = CoFiMethod(
        cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics,
        reference_metric=reference_metric,
    )
    method.setup_model()
    method.setup_data()
    summary = method.fit()
    if evaluate_now:
        summary["metrics"] = method.evaluate()
    summary["trainer"] = method
    return summary


def train_cofi(
    config: Optional[Any] = None,
    model: Any = None,
    tokenizer: Any = None,
    train_dataloader: Any = None,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable] = None,
    reference_metric: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train the **Prune+Distill** (CoFi) baseline (full LM tuned)."""
    kwargs.setdefault("mode", "cofi")
    return _train_one(config, model, tokenizer, train_dataloader, eval_dataloader,
                      compute_metrics, reference_metric, **kwargs)


def train_lora_prune_distill(
    config: Optional[Any] = None,
    model: Any = None,
    tokenizer: Any = None,
    train_dataloader: Any = None,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable] = None,
    reference_metric: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train the **LoRA+Prune+Distill** baseline (only L0 modules + LoRA tunable).

    Follows Sec. 5.2: "a simple baseline is to conduct CoFi pruning and
    distillation but with LoRA parameters tuned only.  More specifically, only
    the :math:`L_0` module and LoRA parameters are tunable".
    """
    kwargs.setdefault("mode", "lora_prune_distill")
    return _train_one(config, model, tokenizer, train_dataloader, eval_dataloader,
                      compute_metrics, reference_metric, **kwargs)


def evaluate_cofi(
    source: Any,
    dataloader: Any = None,
    *,
    task: str = "sst2",
    tokenizer: Any = None,
    device: Optional[str] = None,
    compute_metrics: Optional[Callable] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a :class:`CoFiMethod`, a config, or an already-pruned model."""
    if isinstance(source, CoFiMethod):
        method = source
        metrics = method.evaluate(dataloader)
        out = {"metrics": metrics, "primary": _primary(metrics, method.config.task)}
    else:
        cfg = CoFiConfig.from_dict({"task": task, "device": device} if device else {"task": task})
        config = source if isinstance(source, CoFiConfig) else cfg
        model = source if not isinstance(source, CoFiConfig) else None
        if evaluate_model is not None and model is not None:
            try:
                result = evaluate_model(model, dataloader, task=config.task,
                                        tokenizer=tokenizer, device=device)
                out = {"metrics": result.metrics, "primary": result.primary}
            except Exception:
                out = {"metrics": _evaluate_plain(model, dataloader, config, compute_metrics, tokenizer)}
        else:  # pragma: no cover
            out = {}
        out.setdefault("primary", _primary(out.get("metrics", {}), config.task))
    if save_path:
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# aliases used by the baseline registry
# ---------------------------------------------------------------------------

cofi_trainer = CoFiMethod
LoRAPruneDistillMethod = CoFiMethod


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:  # pragma: no cover - manual sanity check
    ok = True

    # 1. config normalisation
    cfg = CoFiConfig.from_dict({"task": "SST-2", "mode": "LoRA+Prune+Distill"})
    assert cfg.task == "sst2", cfg.task
    assert cfg.mode == "lora_prune_distill", cfg.mode
    assert cfg.display_name == "LoRA+Prune+Distill", cfg.display_name
    assert cfg.tuning_only_lora
    assert abs(cfg.resolved_density - 0.4) < 1e-9

    # 2. Table 6 defaults
    cfg2 = apply_table6_defaults(CoFiConfig.from_dict({"task": "cnndm", "model_type": "t5"}))
    assert cfg2.batch_size == 16 and cfg2.epochs == 16 and cfg2.distill_epochs == 6, cfg2.to_dict()
    assert cfg2.max_seq_length == 512

    # 3. LoRA target modules
    assert lora_target_modules_for("roberta") == ("query", "value")
    assert lora_target_modules_for("t5") == ("q", "v")

    # 4. reference values present
    assert TABLE2_REFERENCES["prune_distill"]["sst2"] == 94.5
    assert TABLE2_REFERENCES["lora_prune_distill"]["mnli"] == 84.2

    if torch is not None:
        # 5. L0 gate behaviour
        gate = L0Gate(8, alpha_init=4.0)
        hard = gate(sample=False)
        assert int(hard.sum().item()) == 8, hard
        gate_zero = L0Gate(8, alpha_init=-4.0)
        assert int(gate_zero(sample=False).sum().item()) == 0, gate_zero(sample=False)

        # 6. gate bank on a tiny wrapped model
        class _Cfg:
            hidden_size = 16
            num_hidden_layers = 2
            num_attention_heads = 4
            intermediate_size = 32

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(16, 16)
                self.value = nn.Linear(16, 16)
                self.dense = nn.Linear(16, 32)
                self.out = nn.Linear(32, 16)

            def forward(self, x):
                return self.out(self.dense(self.query(x) + self.value(x)))

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Cfg()
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList([_Block(), _Block()])

            def forward(self, x):
                for layer in self.encoder.layer:
                    x = layer(x)
                return x

        bank = GateBank(CoFiConfig.from_dict({}))
        bank._add_gate(HEAD, 0, 4, "query")
        bank._add_gate(NEURON, 0, 8, "ffn")
        bank._add_gate(DIMENSION, -1, 16, "hidden")
        masks = bank.sample_masks()
        assert HEAD in masks and NEURON in masks and DIMENSION in masks
        assert masks[DIMENSION].numel() == 16
        assert 0.0 <= bank.sparsity() <= 1.0
        assert bank.expected_l0().item() >= 0.0
        bank.set_temperature(0.1)
        assert abs(bank.temperature_for_progress(1.0) - 0.1) < 1e-9
        units = bank.unit_counts()
        assert units["head"] == 4 and units["neuron"] == 8 and units["dimension"] == 16

        # 7. layer-wise distillation mapping / sampling
        dist = LayerwiseDistillation(CoFiConfig.from_dict({}))
        dist.n_teacher_layers = 4
        dist.n_student_layers = 4
        mapping = dist.update_mapping([1, 1, 1, 1])
        assert mapping == [0, 1, 2, 3], mapping
        mapping = dist.update_mapping([1, 0, 0, 1])
        assert mapping == [0, 0, 0, 3], mapping
        idx = dist.sample_teacher_layers(random.Random(0))
        assert len(idx) == 4 and all(0 <= i < 4 for i in idx), idx

        # 8. distillation forward on synthetic states
        s_states = {0: torch.randn(2, 3, 8), 1: torch.randn(2, 3, 8)}
        t_states = {0: torch.randn(2, 3, 8), 1: torch.randn(2, 3, 8)}
        dist._mapping = [0, 1]
        loss = dist(s_states, t_states, [0, 1])
        assert loss.ndim == 0 and float(loss) >= 0.0
        total = dist.total_loss(torch.tensor(1.0), torch.tensor(2.0), progress=0.5)
        assert float(total) >= 0.0

        # 9. harden-at-sparsity selects a valid keep count
        tiny = GateBank(CoFiConfig.from_dict({}))
        tiny._add_gate(HEAD, 0, 10, "query")
        method = CoFiMethod(CoFiConfig.from_dict({"task": "sst2"}))
        method.gates = tiny
        masks = method._harden_to_sparsity(0.6)
        assert int((masks[HEAD] > 0.5).sum().item()) == 4, masks[HEAD]

    # 10. helper predicates
    assert table6_group_for("roberta", "mnli") == "glue-big"
    assert table6_group_for("roberta", "rte") == "glue-small"
    assert table6_group_for("roberta", "squad_v2") == "squad"
    assert is_seq2seq_task("cnn_dailymail")
    print("apt.baselines.cofi self-test passed")
    return ok


if __name__ == "__main__":  # pragma: no cover
    _self_test()
