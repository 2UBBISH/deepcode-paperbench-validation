#!/usr/bin/env python
"""Train the forgetting forecasters of `What Will My Model Forget?`.

This script trains the two trainable forecasters described in the paper:

* **Logit-change-transfer** forecaster (Sec. 3.2, Eq. 2 / Eq. 3, Algorithms 1 & 2):
  a trainable kernel ``Theta_tilde(x_j, x_i) = h(x_j,y_j) h(x_i,y_i)^T`` predicts the
  logit change of an upstream example ``x_j`` from the observed logit change of the
  online example ``x_i``; optimized with the margin loss of Eq. 3.
  The ``--fixed`` flag switches to the *Fixed Logit* variant of Sec. 4.2 (the kernel is
  the frozen base-PTLM final-layer representation, which is exact when only heads are
  tuned).

* **Representation-based** forecaster (Sec. 3.3, Eq. 4, Algorithms 3 & 4):
  ``z_tilde_ij = sigma(<h(x_j,y_j), h(x_i,y_i)> + b_j)`` with mean-pooled ``h`` and the
  cached frequency prior ``b_j``; optimized with binary cross entropy.
  ``--no-prior`` drops ``b_j`` and yields the paper's ``w/o Prior`` ablation.

Training hyper-parameters reproduce Appendix B ("Training Details of the Forecasting
Models"):

* encoder ``h`` = base LM + freshly initialized 2-layer MLP;
* LM components optimized with lr ``1e-5``, MLP with lr ``1e-4``;
* maximum of ``100,000`` steps with batch size ``16``;
* each mini-batch contains ``8`` positive pairs (``x_j`` forgotten after learning on
  ``<x_i, y_i>``) and ``8`` negative pairs;
* positive pairs are down-weighted by ``alpha = 0.1`` because ground-truth forgetting is
  heavily skewed towards the negative class.

Label convention
----------------
Ground-truth labels are produced by ``src/forgetting/ground_truth.py`` and follow the
Sec. 2 definition ``z_ij = 1[f_i(x_j) != y_j]`` (upstream example forgotten).  Appendix F's
``1[f_0(x_i) != f_i(x_i)]`` is treated as a typo there (documented in that module) and is
therefore never used here.

Artifacts
---------
Inputs are the ground-truth artifacts written by ``scripts/generate_ground_truth.py``::

    <gt_dir>/pairs_train.jsonl      # labelled (i, j, z) pairs + cached top-k logits
    <gt_dir>/pairs_test.jsonl       # held-out pairs (used for periodic evaluation)
    <gt_dir>/online_train.jsonl     # online examples keyed by `index`
    <gt_dir>/frequency_prior.json   # cached b_j

Checkpoints/labels are written to ``<output_dir>/<model_key>[/<tuning>]/`` under the names
expected by ``scripts/forecast_forgetting.py`` (``representation_forecaster.pt`` /
``logit_forecaster.pt``); a copy is also placed in the method sub-directory so several
resolvers find it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# path setup (allows running this file directly, e.g. `python scripts/train_forecaster.py`)
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("train_forecaster")

DEFAULT_CONFIG_PATH = os.path.join(_REPO, "config", "config.yaml")
DEFAULT_GT_DIRNAME = "ground_truth"
DEFAULT_TRAIN_RATIO = 0.6
REPRESENTATION_FILENAME = "representation_forecaster.pt"
LOGIT_FILENAME = "logit_forecaster.pt"
DEFAULT_FORECASTER_STEPS = 100000
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_POSITIVE = 8
DEFAULT_N_NEGATIVE = 8
DEFAULT_POSITIVE_WEIGHT = 0.1
METHODS = ("representation", "logit")


# ======================================================================================
# generic helpers
# ======================================================================================
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML config; return an empty dict when unavailable."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml  # local import so the module stays importable without pyyaml
    except Exception:  # pragma: no cover - pyyaml is a hard requirement in practice
        logger.warning("pyyaml unavailable; using empty config")
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except FileNotFoundError:
        logger.warning("config not found at %s; using empty config", cfg_path)
        return {}
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to parse config %s (%s); using empty config", cfg_path, exc)
        return {}


def cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup: ``cfg_get(cfg, "refinement", "lora", "r", default=16)``."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSON at %s:%d", path, line_no)
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def write_json(path: str, payload: Any) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default_serializer)
    return path


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), default=_default_serializer) + "\n")


def _default_serializer(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def artifact_root(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    base = cfg_get(config, "output_dir", default="artifacts") or "artifacts"
    if not os.path.isabs(base):
        base = os.path.join(_REPO, base)
    parts = [base, model_key]
    if tuning and tuning not in ("none", "None", ""):
        parts.append(str(tuning))
    return os.path.join(*parts)


def dataset_paths(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> Dict[str, str]:
    root = artifact_root(config, model_key, tuning)
    return {
        "root": root,
        "d_pt": os.path.join(root, "d_pt.jsonl"),
        "d_pt_hat": os.path.join(root, "d_pt_hat.jsonl"),
        "d_r": os.path.join(root, "d_r.jsonl"),
        "d_r_train": os.path.join(root, "d_r_train.jsonl"),
        "d_r_test": os.path.join(root, "d_r_test.jsonl"),
        "gt_dir": os.path.join(root, DEFAULT_GT_DIRNAME),
        "cache_dir": cfg_get(config, "cache_dir", default=os.path.join(root, "caches"))
        if not os.path.isabs(str(cfg_get(config, "cache_dir", default="")))
        else cfg_get(config, "cache_dir", default=""),
    }


# --------------------------------------------------------------------------------------
# record accessors (tolerant to dict- or attribute-based records)
# --------------------------------------------------------------------------------------
def rec_get(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return default


def pair_indices(record: Any) -> Tuple[Optional[int], Optional[int]]:
    i = rec_get(record, "i", "online_index", "i_index", default=None)
    j = rec_get(record, "j", "upstream_index", "j_index", default=None)
    try:
        i = int(i) if i is not None else None
    except (TypeError, ValueError):
        i = None
    try:
        j = int(j) if j is not None else None
    except (TypeError, ValueError):
        j = None
    return i, j


def pair_label(record: Any) -> Optional[int]:
    z = rec_get(record, "z", "label", "z_ij", default=None)
    if z is None:
        return None
    try:
        return int(z)
    except (TypeError, ValueError):
        return None


def _tolist(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return value


def topk_arrays(value: Any) -> Optional[Tuple[List[List[int]], List[List[float]]]]:
    """Extract ``(indices, values)`` nested lists from a TopKLogits-like container."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        indices = value.get("indices")
        values = value.get("values")
    else:
        indices = getattr(value, "indices", None)
        values = getattr(value, "values", None)
    indices = _tolist(indices)
    values = _tolist(values)
    if indices is None or values is None:
        return None
    norm_idx: List[List[int]] = []
    norm_val: List[List[float]] = []
    for row in indices:
        row = _tolist(row) or []
        norm_idx.append([int(v) for v in row])
    for row in values:
        row = _tolist(row) or []
        norm_val.append([float(v) for v in row])
    if len(norm_val) != len(norm_idx):
        return None
    return norm_idx, norm_val


# --------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------
def build_index(examples: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    """Index examples by their ``index`` (or ``idx``/``id`` when numeric) field."""
    index: Dict[int, Mapping[str, Any]] = {}
    for pos, ex in enumerate(examples):
        key = None
        for name in ("index", "idx"):
            if isinstance(ex, Mapping) and name in ex:
                key = ex[name]
                break
        if key is None:
            key = pos
        try:
            index[int(key)] = ex
        except (TypeError, ValueError):
            index[pos] = ex
    return index


def load_ground_truth_records(path: str, cls_name: str = "PairRecord") -> List[Any]:
    """Load records through ``src.forgetting.ground_truth`` when possible.

    Falls back to raw dictionaries, which the tolerant accessors above understand.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        from src.forgetting.ground_truth import load_ground_truth_jsonl  # local import
    except Exception:
        return load_jsonl(path)
    try:
        return list(load_ground_truth_jsonl(path))
    except Exception as exc:
        logger.warning("ground_truth loader failed on %s (%s); reading raw JSONL", path, exc)
        return load_jsonl(path)


def load_pairs(gt_dir: str, prefer: Sequence[str] = ("pairs_train.jsonl", "pairs.jsonl")) -> List[Any]:
    for name in prefer:
        path = os.path.join(gt_dir, name)
        if os.path.exists(path):
            records = load_ground_truth_records(path)
            if records:
                logger.info("loaded %d pair records from %s", len(records), path)
                return records
    logger.warning("no pair records found in %s", gt_dir)
    return []


def load_online_examples(gt_dir: str, model_key: str, tuning: Optional[str], config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for name in ("online_train.jsonl", "online.jsonl"):
        path = os.path.join(gt_dir, name)
        if os.path.exists(path):
            records = load_ground_truth_records(path)
            if records:
                logger.info("loaded %d online records from %s", len(records), path)
                return records
    # fall back to D_R / D_R - train artifacts
    paths = dataset_paths(config, model_key, tuning)
    for name in ("d_r_train", "d_r", "d_r_test"):
        if os.path.exists(paths[name]):
            return load_jsonl(paths[name])
    return []


def load_upstream_examples(gt_dir: str, model_key: str, tuning: Optional[str], config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    paths = dataset_paths(config, model_key, tuning)
    for key in ("d_pt_hat", "d_pt"):
        if os.path.exists(paths[key]):
            examples = load_jsonl(paths[key])
            if examples:
                logger.info("loaded %d upstream examples from %s", len(examples), paths[key])
                return examples
    # any copy stored alongside the ground truth is acceptable too
    for name in ("d_pt_hat.jsonl", "d_pt.jsonl"):
        path = os.path.join(gt_dir, name)
        if os.path.exists(path):
            return load_jsonl(path)
    return []


def load_prior_for(gt_dir: str, config: Mapping[str, Any], model_key: str, tuning: Optional[str]) -> Any:
    """Load the cached frequency prior b_j (or ``None`` when unavailable)."""
    candidates: List[str] = []
    if gt_dir:
        candidates.append(os.path.join(gt_dir, "frequency_prior.json"))
    paths = dataset_paths(config, model_key, tuning)
    candidates.append(os.path.join(paths["root"], "frequency_prior.json"))
    if paths.get("cache_dir"):
        candidates.append(os.path.join(paths["cache_dir"], "frequency_prior.json"))
    try:
        from src.forgetting.frequency_prior import load_prior  # local import
    except Exception:
        load_prior = None  # type: ignore
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        if load_prior is not None:
            try:
                prior = load_prior(path)
                logger.info("loaded frequency prior from %s", path)
                return prior
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to load prior %s (%s)", path, exc)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------------------
class PairSampler:
    """Class-balanced sampler: ``n_positive`` forgotten + ``n_negative`` kept pairs.

    Implements the Appendix B recipe of sampling 8 positive and 8 negative pairs per
    mini-batch.  When a class is exhausted the sampler resamples with replacement so the
    fixed mini-batch size is always met (ground-truth forgetting prevalence is only
    ~1-10%, hence positives are the scarce class).
    """

    def __init__(
        self,
        records: Sequence[Any],
        n_positive: int = DEFAULT_N_POSITIVE,
        n_negative: int = DEFAULT_N_NEGATIVE,
        seed: int = 42,
        with_replacement: bool = True,
    ) -> None:
        self.n_positive = int(n_positive)
        self.n_negative = int(n_negative)
        self.with_replacement = bool(with_replacement)
        self.rng = random.Random(seed)
        self.positives: List[Any] = []
        self.negatives: List[Any] = []
        for rec in records:
            z = pair_label(rec)
            if z is None:
                continue
            (self.positives if z == 1 else self.negatives).append(rec)
        if not self.positives:
            logger.warning("no positive pairs available; positive class will be replaced")
        if not self.negatives:
            logger.warning("no negative pairs available; negative class will be replaced")

    def __len__(self) -> int:
        return len(self.positives) + len(self.negatives)

    @property
    def positive_rate(self) -> float:
        total = len(self.positives) + len(self.negatives)
        return (len(self.positives) / total) if total else 0.0

    def _draw(self, pool: Sequence[Any], n: int) -> List[Any]:
        if not pool:
            return []
        if n <= 0:
            return []
        if self.with_replacement or n > len(pool):
            return [pool[self.rng.randrange(len(pool))] for _ in range(n)]
        return self.rng.sample(list(pool), n)

    def sample(self) -> List[Any]:
        batch = self._draw(self.positives, self.n_positive) + self._draw(self.negatives, self.n_negative)
        self.rng.shuffle(batch)
        return batch


def unique_with_reverse(indices: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Return unique indices and the position of each original index in the unique list."""
    order: Dict[int, int] = {}
    uniq: List[int] = []
    reverse: List[int] = []
    for idx in indices:
        if idx not in order:
            order[idx] = len(uniq)
            uniq.append(idx)
        reverse.append(order[idx])
    return uniq, reverse


# --------------------------------------------------------------------------------------
# encoder / forecaster construction
# --------------------------------------------------------------------------------------
def resolve_encoder(
    model_key: str,
    config: Mapping[str, Any],
    device: str,
    dtype: Any = None,
    freeze_backbone: bool = False,
    checkpoint: Optional[str] = None,
) -> Any:
    """Build the trainable encoding function ``h`` (base LM + 2-layer MLP)."""
    from src.modeling.encoder_h import load_encoder_h

    encoder_cfg = dict(cfg_get(config, "encoder_h", default={}) or {})
    overrides: Dict[str, Any] = {"device": device}
    if dtype is not None:
        overrides["dtype"] = dtype
    if freeze_backbone:
        overrides["freeze_backbone"] = True
    for key in ("mlp_hidden", "mlp_layers", "repr_dim", "token_dim", "lm_lr", "mlp_lr", "dropout"):
        if key in encoder_cfg:
            overrides.setdefault(key, encoder_cfg[key])
    if "dim" not in overrides and "repr_dim" in overrides:
        overrides["dim"] = overrides["repr_dim"]
    encoder = load_encoder_h(model_key, config=config, **overrides)
    if checkpoint and os.path.exists(checkpoint):
        try:
            if hasattr(encoder, "load_state"):
                encoder.load_state(checkpoint)
                logger.info("loaded encoder state from %s", checkpoint)
        except Exception as exc:  # pragma: no cover
            logger.warning("could not load encoder state from %s (%s)", checkpoint, exc)
    return encoder


def build_forecaster(
    method: str,
    model_key: str,
    config: Mapping[str, Any],
    device: str,
    dtype: Any = None,
    fixed: bool = False,
    use_prior: bool = True,
    prior: Any = None,
    positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
    topk: int = 100,
    checkpoint: Optional[str] = None,
) -> Any:
    encoder = resolve_encoder(
        model_key, config, device, dtype=dtype, freeze_backbone=bool(fixed and method == "logit")
    )
    dim = int(
        cfg_get(config, "encoder_h", "repr_dim", default=None)
        or cfg_get(config, "encoder_h", "mlp_hidden", default=768)
        or 768
    )
    if method == "representation":
        from src.forecasters.representation_based import RepresentationBasedForecaster

        forecaster = RepresentationBasedForecaster(
            encoder=encoder,
            dim=dim,
            use_prior=bool(use_prior),
            positive_weight=float(positive_weight),
            prior=prior,
        )
    elif method == "logit":
        if fixed:
            from src.forecasters.logit_based import FixedLogitForecaster

            forecaster = FixedLogitForecaster(encoder=encoder, dim=dim)
        else:
            from src.forecasters.logit_based import LogitChangeTransferForecaster

            forecaster = LogitChangeTransferForecaster(encoder=encoder, dim=dim, topk=int(topk))
    else:
        raise ValueError("unknown forecaster method: %r" % (method,))

    if checkpoint and os.path.exists(checkpoint):
        try:
            if hasattr(forecaster, "load_state"):
                forecaster.load_state(checkpoint, strict=False)
                logger.info("resumed forecaster state from %s", checkpoint)
        except Exception as exc:  # pragma: no cover
            logger.warning("could not load forecaster state from %s (%s)", checkpoint, exc)
    if use_prior and prior is not None and hasattr(forecaster, "set_prior"):
        try:
            forecaster.set_prior(prior)
        except Exception as exc:  # pragma: no cover
            logger.warning("could not attach prior to forecaster (%s)", exc)
    return forecaster


def trainable_parameters(module: Any) -> List[Any]:
    for name in ("trainable_parameters", "parameters"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                params = fn()
                params = [p for p in params if getattr(p, "requires_grad", False)]
                if params:
                    return params
            except Exception:
                continue
    return [p for p in getattr(module, "parameters", lambda: [])() if getattr(p, "requires_grad", False)]


def build_optimizer(
    module: Any,
    lr_lm: float,
    lr_mlp: float,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
):
    """AdamW with per-component learning rates (LM ``1e-5``, MLP ``1e-4``, Appendix B)."""
    import torch

    groups: Optional[List[Dict[str, Any]]] = None
    for name in ("param_groups",):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                candidate = fn()
            except TypeError:
                candidate = None
            except Exception:
                candidate = None
            if candidate:
                groups = []
                for group in candidate:
                    if not isinstance(group, Mapping):
                        continue
                    params = [p for p in group.get("params", []) if getattr(p, "requires_grad", False)]
                    if not params:
                        continue
                    lr = group.get("lr", None)
                    if lr is None:
                        lr = lr_lm
                    groups.append({"params": params, "lr": float(lr)})
                if groups:
                    break
                groups = None
    if not groups:
        params = trainable_parameters(module)
        groups = [{"params": params, "lr": float(lr_mlp)}]
        logger.info("optimizer: single param group (lr=%g) over %d tensors", lr_mlp, len(params))
    return torch.optim.AdamW(groups, lr=float(lr_mlp), betas=tuple(betas), weight_decay=float(weight_decay))


# --------------------------------------------------------------------------------------
# encoding helpers
# --------------------------------------------------------------------------------------
def _as_tensor(value: Any):
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("hidden", "hidden_states", "representations", "embeddings", "last_hidden_state"):
            if key in value:
                return _as_tensor(value[key])
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(value)
        except Exception:
            return None
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def encode_examples(
    module: Any,
    inputs: Sequence[str],
    targets: Sequence[Optional[str]],
    mean_pool: bool = True,
    batch_size: int = 8,
):
    """Encode ``(x, y)`` pairs with a tolerant adapter over the ``h`` implementations."""
    if not inputs:
        import torch

        return torch.zeros(0)
    fn = getattr(module, "encode", None)
    attempts: List[Any] = []
    if mean_pool:
        attempts = [
            lambda: fn(inputs, targets, mean_pool=True, batch_size=batch_size),
            lambda: fn(inputs, targets, mean_pool=True),
            lambda: getattr(module, "encode_mean")(inputs, targets, batch_size=batch_size),
            lambda: getattr(module, "encode_mean")(inputs, targets),
            lambda: fn(list(inputs), list(targets)),
        ]
    else:
        attempts = [
            lambda: fn(inputs, targets, token_level=True, batch_size=batch_size),
            lambda: fn(inputs, targets, mean_pool=False, batch_size=batch_size),
            lambda: fn(inputs, targets, mean_pool=False),
            lambda: getattr(module, "encode_token_level")(inputs, targets, batch_size=batch_size),
            lambda: getattr(module, "encode_token_level")(inputs, targets),
            lambda: fn(list(inputs), list(targets)),
        ]
    last_err: Optional[Exception] = None
    for attempt in attempts:
        try:
            out = attempt()
        except AttributeError as exc:
            last_err = exc
            continue
        except TypeError as exc:
            last_err = exc
            continue
        except Exception as exc:  # real failure, keep trying other signatures
            last_err = exc
            continue
        tensor = _as_tensor(out)
        if tensor is not None:
            return tensor
    raise RuntimeError("could not encode examples with %r (%s)" % (type(module).__name__, last_err))


def find_tokenizer(module: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Locate a tokenizer (and its owner object) inside a forecaster/encoder/backbone."""
    seen: List[Any] = [module]
    for _ in range(3):
        nxt: List[Any] = []
        for obj in seen:
            if obj is None:
                continue
            tok = getattr(obj, "tokenizer", None)
            if tok is not None:
                return tok, obj
            for attr in ("backbone", "encoder", "base_lm", "model"):
                child = getattr(obj, attr, None)
                if child is not None and child is not obj:
                    nxt.append(child)
        seen = nxt
        if not seen:
            break
    return None, None


def target_ids_for(module: Any, targets: Sequence[Optional[str]], max_len: int = 64):
    """Tokenize gold outputs into ids of shape ``[B, T]`` with ``-100`` padding."""
    import torch

    tok, owner = find_tokenizer(module)
    if tok is None:
        raise RuntimeError(
            "no tokenizer found for target tokenization; the logit forecaster needs gold "
            "token ids for the Eq. 3 margin loss"
        )
    texts = ["" if t is None else str(t) for t in targets]
    ids = None
    if owner is not None and hasattr(owner, "tokenize_targets"):
        try:
            out = owner.tokenize_targets(list(texts))
            ids = out.get("input_ids") if isinstance(out, Mapping) else out
        except Exception:
            ids = None
    if ids is None:
        try:
            enc = tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"]
        except Exception as exc:
            raise RuntimeError("failed to tokenize target strings: %s" % (exc,))
    ids = _as_tensor(ids)
    if ids is None:
        raise RuntimeError("tokenizer returned unusable target ids")
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    pad_id = getattr(tok, "pad_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    bos_ids = {v for v in (getattr(tok, "bos_token_id", None), getattr(tok, "decoder_start_token_id", None)) if v is not None}

    out = ids.clone()
    if bos_ids:
        first = out[:, 0]
        strip = torch.zeros_like(first, dtype=torch.bool)
        for bos in bos_ids:
            strip = strip | (first == int(bos))
        if bool(strip.any()):
            out = torch.cat([out[:, 1:], out.new_full((out.size(0), 1), pad_id if pad_id is not None else 0)], dim=1)
    if pad_id is not None:
        out = out.masked_fill(out == int(pad_id), -100)
    if eos_id is not None:
        out = out.masked_fill(out == int(eos_id), -100)
    return out


# --------------------------------------------------------------------------------------
# candidate-space helpers for the logit forecaster (top-k cache -> dense tensors)
# --------------------------------------------------------------------------------------
def candidates_from(topk_list: Sequence[Any], target_ids=None, max_candidates: Optional[int] = None) -> List[int]:
    """Union of cached top-k ids (+ gold ids), mirroring ``logit_based.build_candidate_indices``."""
    try:
        from src.forecasters import logit_based as LB  # local import

        fn = getattr(LB, "build_candidate_indices", None)
        if callable(fn):
            indices = [arr for arr in topk_list if arr is not None]
            try:
                out = fn(indices, target_ids=target_ids, max_candidates=max_candidates)
                out = _tolist(out)
                if out:
                    return [int(v) for v in out]
            except Exception:
                pass
            for arr in indices:
                try:
                    out = fn(arr, target_ids=target_ids, max_candidates=max_candidates)
                    out = _tolist(out)
                    if out:
                        return [int(v) for v in out]
                except Exception:
                    continue
    except Exception:
        pass

    cand = set()
    for arr in topk_list:
        if not arr:
            continue
        for row in arr:
            for v in row or []:
                cand.add(int(v))
    if target_ids is not None:
        flat = _tolist(target_ids) or []
        for row in flat if any(isinstance(r, list) for r in flat) else [flat]:
            for v in row or []:
                if v is not None and int(v) >= 0:
                    cand.add(int(v))
    out = sorted(cand)
    if max_candidates:
        out = out[: int(max_candidates)]
    return out


def densify_topk(indices_rows: Sequence[Sequence[int]], values_rows: Sequence[Sequence[float]], candidates: Sequence[int], fill_value: float = -1e4):
    """Scatter cached top-k logits onto the ``[T, |candidates|]`` dense candidate space."""
    import torch

    dtype = torch.float32
    try:
        from src.forecasters import logit_based as LB  # local import

        fn = getattr(LB, "densify_topk", None)
        if callable(fn):
            try:
                return fn(indices_rows, values_rows, candidates, fill_value=fill_value, dtype=dtype)
            except TypeError:
                return fn(indices_rows, values_rows, candidates)
    except Exception:
        pass

    pos = {int(v): k for k, v in enumerate(candidates)}
    T = len(indices_rows)
    S = len(candidates)
    out = torch.full((T, S), float(fill_value), dtype=dtype)
    for t in range(T):
        row_i = indices_rows[t] if t < len(indices_rows) else []
        row_v = values_rows[t] if t < len(values_rows) else []
        for v_idx, v_val in zip(row_i, row_v):
            k = pos.get(int(v_idx))
            if k is not None:
                out[t, k] = float(v_val)
    return out


def delta_matrix(f0_indices, f0_values, fi_indices, fi_values, candidates):
    """``f_i(x_i) - f_0(x_i)`` over the candidate space (missing entries => 0 change)."""
    import torch

    try:
        from src.forecasters import logit_based as LB  # local import

        fn = getattr(LB, "make_delta_matrix", None)
        if callable(fn):
            try:
                return fn(f0_indices, f0_values, fi_indices, fi_values, candidates, zero_missing=True, dtype=torch.float32)
            except TypeError:
                return fn(f0_indices, f0_values, fi_indices, fi_values, candidates)
    except Exception:
        pass

    f0_dense = densify_topk(f0_indices, f0_values, candidates, fill_value=0.0)
    fi_dense = densify_topk(fi_indices, fi_values, candidates, fill_value=0.0)
    return fi_dense - f0_dense


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def _target_text(example: Any, default: str = "") -> str:
    t = rec_get(example, "target", "targets", "label", "reference", default=None)
    if isinstance(t, (list, tuple)):
        t = t[0] if t else None
    return "" if t is None else str(t)


def train_representation(
    forecaster: Any,
    sampler: PairSampler,
    online_by_index: Mapping[int, Any],
    upstream: Sequence[Any],
    prior: Any,
    max_steps: int,
    batch_size: int,
    lr_lm: float,
    lr_mlp: float,
    positive_weight: float,
    device: str,
    seed: int,
    log_every: int = 1000,
    save_every: int = 0,
    save_path: Optional[str] = None,
    log_path: Optional[str] = None,
    eval_fn: Optional[Any] = None,
    eval_every: int = 0,
    max_grad_norm: float = 1.0,
    start_step: int = 0,
) -> Dict[str, Any]:
    """Train the Eq. 4 representation-based forecaster (Algorithm 3)."""
    import torch

    optimizer = build_optimizer(forecaster, lr_lm=lr_lm, lr_mlp=lr_mlp)
    forecaster.train()
    history: List[Dict[str, Any]] = []
    running: List[float] = []
    t0 = time.time()
    rng = random.Random(seed + 1)

    for step in range(start_step, max_steps):
        batch = sampler.sample()
        pairs = [(pair_indices(rec), pair_label(rec), rec) for rec in batch]
        pairs = [(ij, z, rec) for (ij, z, rec) in pairs if ij[0] is not None and ij[1] is not None and z is not None]
        if not pairs:
            continue

        uni_i, rev_i = unique_with_reverse([ij[0] for ij, _, _ in pairs])
        uni_j, rev_j = unique_with_reverse([ij[1] for ij, _, _ in pairs])

        in_texts: List[str] = []
        tg_texts: List[Optional[str]] = []
        for idx in uni_i:
            ex = online_by_index.get(int(idx))
            if ex is None:
                in_texts.append("")
                tg_texts.append("")
            else:
                in_texts.append(str(rec_get(ex, "input", "inputs", "prompt", default="")))
                tg_texts.append(_target_text(ex))
        h_online_all = encode_examples(forecaster, in_texts, tg_texts, mean_pool=True, batch_size=batch_size)

        up_texts: List[str] = []
        up_targets: List[Optional[str]] = []
        for idx in uni_j:
            ex = upstream[int(idx)] if 0 <= int(idx) < len(upstream) else None
            up_texts.append("" if ex is None else str(rec_get(ex, "input", "inputs", "prompt", default="")))
            up_targets.append(None if ex is None else _target_text(ex))
        h_upstream_all = encode_examples(forecaster, up_texts, up_targets, mean_pool=True, batch_size=batch_size)

        idx_i = torch.tensor(rev_i, dtype=torch.long, device=h_online_all.device)
        idx_j = torch.tensor(rev_j, dtype=torch.long, device=h_upstream_all.device)
        h_online = h_online_all.index_select(0, idx_i) if h_online_all.dim() > 1 else h_online_all
        h_upstream = h_upstream_all.index_select(0, idx_j) if h_upstream_all.dim() > 1 else h_upstream_all

        z = torch.tensor([int(z) for _, z, _ in pairs], dtype=torch.float32, device=h_online.device)
        prior_vec = None
        if prior is not None and getattr(forecaster, "use_prior", True):
            values = []
            for (_, j), _, _ in pairs:
                values.append(prior_lookup(prior, int(j)))
            prior_vec = torch.tensor(values, dtype=torch.float32, device=h_online.device)

        try:
            loss = forecaster.loss(h_upstream, h_online, z, prior=prior_vec)
        except TypeError:
            loss = forecaster.loss(h_upstream, h_online, z)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(forecaster), float(max_grad_norm))
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        running.append(loss_value)

        if log_every and (step + 1) % log_every == 0:
            mean_loss = sum(running) / max(1, len(running))
            entry = {
                "step": step + 1,
                "loss": mean_loss,
                "elapsed_sec": time.time() - t0,
                "positive_weight": positive_weight,
            }
            logger.info("step %d/%d loss=%.4f (%.1fs)", step + 1, max_steps, mean_loss, entry["elapsed_sec"])
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                try:
                    metrics = eval_fn(forecaster)
                    entry.update({("eval_" + k): v for k, v in (metrics or {}).items()})
                    logger.info("  eval @%d: %s", step + 1, metrics)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  evaluation failed: %s", exc)
            history.append(entry)
            if log_path:
                append_jsonl(log_path, entry)
            running = []

        if save_path and save_every and (step + 1) % save_every == 0:
            save_forecaster(forecaster, save_path, extra={"step": step + 1, "method": "representation"})

    if save_path:
        save_forecaster(forecaster, save_path, extra={"step": max_steps, "method": "representation"})
    return {
        "steps": max_steps,
        "final_loss": (sum(running) / len(running)) if running else (history[-1]["loss"] if history else None),
        "history": history,
        "elapsed_sec": time.time() - t0,
    }


def train_logit(
    forecaster: Any,
    sampler: PairSampler,
    online_by_index: Mapping[int, Any],
    upstream: Sequence[Any],
    max_steps: int,
    batch_size: int,
    lr_lm: float,
    lr_mlp: float,
    positive_weight: float,
    device: str,
    seed: int,
    topk: int = 100,
    log_every: int = 1000,
    save_every: int = 0,
    save_path: Optional[str] = None,
    log_path: Optional[str] = None,
    eval_fn: Optional[Any] = None,
    eval_every: int = 0,
    max_grad_norm: float = 1.0,
    margin: float = 1.0,
    max_target_len: int = 64,
    start_step: int = 0,
) -> Dict[str, Any]:
    """Train the Eq. 2/Eq. 3 logit-change-transfer forecaster (Algorithm 1).

    The Eq. 3 margin loss is computed for every pair individually (the kernel lives in
    ``R^{T_j x T_i}`` and ``T_j`` differs across upstream examples); gradients are
    accumulated so that the effective batch size still equals ``batch_size`` (16).
    """
    import torch
    from src.forecasters import losses as L  # local import

    optimizer = build_optimizer(forecaster, lr_lm=lr_lm, lr_mlp=lr_mlp)
    forecaster.train()
    history: List[Dict[str, Any]] = []
    running: List[float] = []
    skipped = 0
    t0 = time.time()

    def _pair_tensors(rec: Any, j_index: int, j_target_text: str, i_index: int, i_target_text: str):
        """Build ``(h_j, h_i, delta, f0_xj, candidates, target_ids)`` for one pair."""
        i_logits = rec_get(rec, "f0_i_token_logits", "f0_xi_token_logits", default=None)
        i_logits_fi = rec_get(rec, "fi_i_token_logits", "fi_xi_token_logits", default=None)
        if i_logits is None or i_logits_fi is None:
            online_rec = online_by_index.get(int(i_index))
            if online_rec is not None:
                i_logits = i_logits or rec_get(online_rec, "f0_token_logits", "f0_token_logits_i", default=None)
                i_logits_fi = i_logits_fi or rec_get(online_rec, "fi_token_logits", "fi_token_logits_i", default=None)
        j_logits = rec_get(rec, "f0_j_token_logits", "f0_token_logits_j", default=None)

        f0_i = topk_arrays(i_logits)
        fi_i = topk_arrays(i_logits_fi)
        f0_j = topk_arrays(j_logits)
        if f0_i is None or fi_i is None or f0_j is None:
            return None

        up_ex = upstream[int(j_index)] if 0 <= int(j_index) < len(upstream) else None
        up_in = "" if up_ex is None else str(rec_get(up_ex, "input", "inputs", "prompt", default=""))
        on_ex = online_by_index.get(int(i_index))
        on_in = "" if on_ex is None else str(rec_get(on_ex, "input", "inputs", "prompt", default=""))

        tgt_j = target_ids_for(forecaster, [j_target_text], max_len=max_target_len)
        tgt_i = target_ids_for(forecaster, [i_target_text], max_len=max_target_len)

        cand = candidates_from([f0_i[0], fi_i[0], f0_j[0]], target_ids=tgt_j)
        if not cand:
            return None

        h_j = encode_examples(forecaster, [up_in], [j_target_text], mean_pool=False, batch_size=1)
        h_i = encode_examples(forecaster, [on_in], [i_target_text], mean_pool=False, batch_size=1)
        delta = delta_matrix(f0_i[0], f0_i[1], fi_i[0], fi_i[1], cand)
        f0_xj = densify_topk(f0_j[0], f0_j[1], cand, fill_value=L.NEG_INF)
        return h_j, h_i, delta, f0_xj, cand, tgt_j, tgt_i

    for step in range(start_step, max_steps):
        batch = sampler.sample()
        optimizer.zero_grad(set_to_none=True)
        step_losses: List[float] = []
        for rec in batch:
            i_index, j_index = pair_indices(rec)
            z = pair_label(rec)
            if i_index is None or j_index is None or z is None:
                skipped += 1
                continue
            i_target_text = str(rec_get(rec, "i_target", "online_target", default=""))
            if not i_target_text:
                on_ex = online_by_index.get(int(i_index))
                i_target_text = "" if on_ex is None else _target_text(on_ex)
            j_target_text = str(rec_get(rec, "j_target", "upstream_target", default=""))
            if not j_target_text and 0 <= int(j_index) < len(upstream):
                j_target_text = _target_text(upstream[int(j_index)])

            try:
                tensors = _pair_tensors(rec, int(j_index), j_target_text, int(i_index), i_target_text)
            except Exception as exc:
                logger.debug("pair skipped (%s)", exc)
                tensors = None
            if tensors is None:
                skipped += 1
                continue

            h_j, h_i, delta, f0_xj, cand, tgt_j, _tgt_i = tensors
            try:
                pred = forecaster.forward(h_j, h_i, delta, f0_xj)
            except TypeError:
                try:
                    pred = forecaster.forward(h_j, h_i, delta, f0_xj, candidates=cand, target_ids=tgt_j)
                except Exception as exc:
                    logger.debug("forward failed (%s)", exc)
                    skipped += 1
                    continue

            try:
                pair_loss = forecaster.loss(pred, cand, tgt_j, z, margin=margin)
            except TypeError:
                pair_loss = forecaster.loss(pred, cand, tgt_j, z)
            if z == 1:
                # Appendix B: down-weight positive (forgotten) pairs with alpha = 0.1
                pair_loss = pair_loss * float(positive_weight)
            pair_loss = pair_loss / max(1, len(batch))
            pair_loss.backward()
            step_losses.append(float(pair_loss.detach().cpu().item()) * max(1, len(batch)))

        if not step_losses:
            skipped += 1
            continue
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(forecaster), float(max_grad_norm))
        optimizer.step()

        loss_value = sum(step_losses) / max(1, len(step_losses))
        running.append(loss_value)

        if log_every and (step + 1) % log_every == 0:
            mean_loss = sum(running) / max(1, len(running))
            entry = {
                "step": step + 1,
                "loss": mean_loss,
                "elapsed_sec": time.time() - t0,
                "skipped": skipped,
            }
            logger.info("step %d/%d loss=%.4f skipped=%d (%.1fs)", step + 1, max_steps, mean_loss, skipped, entry["elapsed_sec"])
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                try:
                    metrics = eval_fn(forecaster)
                    entry.update({("eval_" + k): v for k, v in (metrics or {}).items()})
                    logger.info("  eval @%d: %s", step + 1, metrics)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  evaluation failed: %s", exc)
            history.append(entry)
            if log_path:
                append_jsonl(log_path, entry)
            running = []

        if save_path and save_every and (step + 1) % save_every == 0:
            save_forecaster(forecaster, save_path, extra={"step": step + 1, "method": "logit"})

    if save_path:
        save_forecaster(forecaster, save_path, extra={"step": max_steps, "method": "logit"})
    return {
        "steps": max_steps,
        "final_loss": (sum(running) / len(running)) if running else (history[-1]["loss"] if history else None),
        "skipped_pairs": skipped,
        "history": history,
        "elapsed_sec": time.time() - t0,
    }


# --------------------------------------------------------------------------------------
# evaluation during training
# --------------------------------------------------------------------------------------
def prior_lookup(prior: Any, upstream_index: int, default: float = 0.0) -> float:
    try:
        from src.forgetting.frequency_prior import prior_for_upstream  # local import

        try:
            return float(prior_for_upstream(prior, upstream_index, default=default))
        except TypeError:
            return float(prior_for_upstream(prior, upstream_index))
    except Exception:
        pass
    if prior is None:
        return float(default)
    if isinstance(prior, Mapping):
        for key in (upstream_index, str(upstream_index)):
            if key in prior:
                try:
                    return float(prior[key])
                except (TypeError, ValueError):
                    return float(default)
        inner = prior.get("priors")
        if isinstance(inner, Mapping):
            for key in (upstream_index, str(upstream_index)):
                if key in inner:
                    try:
                        return float(inner[key])
                    except (TypeError, ValueError):
                        return float(default)
        return float(prior.get("default", default) or default)
    getter = getattr(prior, "get", None)
    if callable(getter):
        try:
            value = getter(upstream_index, default)
        except TypeError:
            value = getter(upstream_index)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)
    return float(default)


def make_eval_fn(
    method: str,
    test_pairs: Sequence[Any],
    online_by_index: Mapping[int, Any],
    upstream: Sequence[Any],
    prior: Any,
    device: str,
    max_pairs: int = 256,
    max_target_len: int = 64,
) -> Optional[Any]:
    """Build a callable that scores the forecaster on held-out pairs (F1)."""
    if not test_pairs:
        return None
    subset = list(test_pairs[: int(max_pairs)]) if max_pairs else list(test_pairs)

    def _eval(forecaster: Any) -> Dict[str, float]:
        import torch

        was_training = getattr(forecaster, "training", False)
        if hasattr(forecaster, "eval"):
            forecaster.eval()
        z_true: List[int] = []
        z_pred: List[int] = []
        try:
            if method == "representation":
                in_texts, tg_texts, j_list, z_list = [], [], [], []
                for rec in subset:
                    i_index, j_index = pair_indices(rec)
                    z = pair_label(rec)
                    if i_index is None or j_index is None or z is None:
                        continue
                    if not (0 <= int(j_index) < len(upstream)):
                        continue
                    on_ex = online_by_index.get(int(i_index))
                    in_texts.append("" if on_ex is None else str(rec_get(on_ex, "input", "inputs", "prompt", default="")))
                    tg_texts.append("" if on_ex is None else _target_text(on_ex))
                    j_list.append(int(j_index))
                    z_list.append(int(z))
                if not z_list:
                    return {}
                h_online = encode_examples(forecaster, in_texts, tg_texts, mean_pool=True, batch_size=16)
                up_texts = [str(rec_get(upstream[j], "input", "inputs", "prompt", default="")) for j in j_list]
                up_tgts = [_target_text(upstream[j]) for j in j_list]
                h_upstream = encode_examples(forecaster, up_texts, up_tgts, mean_pool=True, batch_size=16)
                prior_vec = None
                if prior is not None and getattr(forecaster, "use_prior", True):
                    prior_vec = torch.tensor([prior_lookup(prior, j) for j in j_list], dtype=torch.float32, device=h_online.device)
                probs = representation_probabilities(forecaster, h_upstream, h_online, prior_vec)
                z_pred = [1 if float(p) >= 0.5 else 0 for p in probs.detach().cpu().reshape(-1)]
                z_true = z_list
            else:
                for rec in subset:
                    i_index, j_index = pair_indices(rec)
                    z = pair_label(rec)
                    if i_index is None or j_index is None or z is None:
                        continue
                    if not (0 <= int(j_index) < len(upstream)):
                        continue
                    i_target_text = "" if online_by_index.get(int(i_index)) is None else _target_text(online_by_index[int(i_index)])
                    j_target_text = _target_text(upstream[int(j_index)])
                    pred = predict_logit_pair(forecaster, rec, i_index, j_index, i_target_text, j_target_text, online_by_index, upstream, max_target_len)
                    if pred is None:
                        continue
                    z_pred.append(int(pred))
                    z_true.append(int(z))
        finally:
            if was_training and hasattr(forecaster, "train"):
                forecaster.train()
        if not z_true:
            return {}
        return {'f1': binary_f1(z_true, z_pred), 'precision': binary_precision(z_true, z_pred), 'recall': binary_recall(z_true, z_pred), 'n': len(z_true)}

    return _eval


def representation_probabilities(forecaster: Any, h_upstream, h_online, prior_vec=None):
    fn = getattr(forecaster, "score", None)
    if callable(fn):
        try:
            logits = fn(h_upstream, h_online, prior=prior_vec)
        except TypeError:
            try:
                logits = fn(h_upstream, h_online)
            except Exception:
                logits = None
        if logits is not None:
            import torch

            return torch.sigmoid(logits.reshape(-1) if logits.dim() > 1 else logits)
    import torch

    score = (h_upstream * h_online).sum(dim=-1)
    if prior_vec is not None:
        score = score + prior_vec
    return torch.sigmoid(score)


def predict_logit_pair(
    forecaster: Any,
    rec: Any,
    i_index: int,
    j_index: int,
    i_target_text: str,
    j_target_text: str,
    online_by_index: Mapping[int, Any],
    upstream: Sequence[Any],
    max_target_len: int = 64,
) -> Optional[int]:
    """``z_hat = 1[argmax_v f_hat_i(x_j)[v] != y_j]`` from cached logits (Algorithm 2)."""
    i_logits = rec_get(rec, "f0_i_token_logits", "f0_xi_token_logits", default=None)
    i_logits_fi = rec_get(rec, "fi_i_token_logits", "fi_xi_token_logits", default=None)
    if i_logits is None or i_logits_fi is None:
        on_rec = online_by_index.get(int(i_index))
        if on_rec is not None:
            i_logits = i_logits or rec_get(on_rec, "f0_token_logits", default=None)
            i_logits_fi = i_logits_fi or rec_get(on_rec, "fi_token_logits", default=None)
    j_logits = rec_get(rec, "f0_j_token_logits", "f0_token_logits_j", default=None)
    f0_i = topk_arrays(i_logits)
    fi_i = topk_arrays(i_logits_fi)
    f0_j = topk_arrays(j_logits)
    if f0_i is None or fi_i is None or f0_j is None:
        return None

    try:
        tgt_j = target_ids_for(forecaster, [j_target_text], max_len=max_target_len)
    except Exception:
        return None
    cand = candidates_from([f0_i[0], fi_i[0], f0_j[0]], target_ids=tgt_j)
    if not cand:
        return None

    up_in = str(rec_get(upstream[int(j_index)], "input", "inputs", "prompt", default=""))
    on_ex = online_by_index.get(int(i_index))
    on_in = "" if on_ex is None else str(rec_get(on_ex, "input", "inputs", "prompt", default=""))

    h_j = encode_examples(forecaster, [up_in], [j_target_text], mean_pool=False, batch_size=1)
    h_i = encode_examples(forecaster, [on_in], [i_target_text], mean_pool=False, batch_size=1)
    delta = delta_matrix(f0_i[0], f0_i[1], fi_i[0], fi_i[1], cand)
    f0_xj = densify_topk(f0_j[0], f0_j[1], cand, fill_value=-1e4)
    try:
        pred = forecaster.forward(h_j, h_i, delta, f0_xj)
    except Exception:
        return None

    import torch

    gold = tgt_j.reshape(-1)
    gold = gold[gold >= 0]
    if gold.numel() == 0:
        return None
    logits = pred.reshape(-1, len(cand)) if pred.dim() > 1 else pred
    top = torch.argmax(logits, dim=-1)
    top_ids = [cand[int(k)] for k in top.detach().cpu().tolist()]
    n = min(len(top_ids), int(gold.numel()))
    if n == 0:
        return None
    mismatch = any(int(top_ids[t]) != int(gold[t]) for t in range(n))
    return 1 if mismatch else 0


# --------------------------------------------------------------------------------------
# metrics (thin wrappers; canonical implementations live in src/eval/metrics.py)
# --------------------------------------------------------------------------------------
def _counts(z_true: Sequence[int], z_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    tp = sum(1 for a, b in zip(z_true, z_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(z_true, z_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(z_true, z_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(z_true, z_pred) if a == 0 and b == 0)
    return tp, fp, fn, tn


def binary_f1(z_true: Sequence[int], z_pred: Sequence[int]) -> float:
    try:
        from src.eval.metrics import binary_f1 as _f1  # local import

        return float(_f1(z_true, z_pred))
    except Exception:
        pass
    tp, fp, fn, _ = _counts(z_true, z_pred)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom else 0.0


def binary_precision(z_true: Sequence[int], z_pred: Sequence[int]) -> float:
    try:
        from src.eval.metrics import precision_score_binary as _p  # local import

        return float(_p(z_true, z_pred))
    except Exception:
        pass
    tp, fp, _, _ = _counts(z_true, z_pred)
    return (tp / (tp + fp)) if (tp + fp) else 0.0


def binary_recall(z_true: Sequence[int], z_pred: Sequence[int]) -> float:
    try:
        from src.eval.metrics import recall_score_binary as _r  # local import

        return float(_r(z_true, z_pred))
    except Exception:
        pass
    tp, _, fn, _ = _counts(z_true, z_pred)
    return (tp / (tp + fn)) if (tp + fn) else 0.0


# --------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------
def save_forecaster(forecaster: Any, path: str, extra: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Persist the forecaster, writing to ``<method>/`` as well for resolver robustness."""
    written: List[str] = []
    targets = [path]
    parent, name = os.path.split(path)
    method = str((extra or {}).get("method", "")) or None
    if method:
        alt = os.path.join(parent, method, name)
        if alt not in targets:
            targets.append(alt)
    for target in targets:
        try:
            parent_dir = os.path.dirname(os.path.abspath(target))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            fn = getattr(forecaster, "save", None)
            if callable(fn):
                try:
                    fn(target, extra=dict(extra or {}))
                except TypeError:
                    fn(target)
            else:
                import torch

                torch.save({"state_dict": forecaster.state_dict(), "extra": dict(extra or {})}, target)
            written.append(target)
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to save checkpoint %s (%s)", target, exc)
    if written:
        logger.info("saved forecaster to %s", ", ".join(written))
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forgetting forecasters (Tables 1-2).")
    parser.add_argument("--method", choices=list(METHODS), default="representation")
    parser.add_argument("--model-key", default="BART0_L")
    parser.add_argument("--tuning", default=None, help="head | lora | full_ft | none")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gt-dir", default=None, help="ground-truth directory (defaults to artifact root)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="checkpoint to resume from")
    parser.add_argument("--no-init-checkpoint", action="store_true", help="ignore --checkpoint for initialisation")
    parser.add_argument("--fixed", action="store_true", help="Fixed Logit variant (frozen representation kernel)")
    parser.add_argument("--no-prior", action="store_true", help="representation ablation without frequency prior")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-positive", type=int, default=DEFAULT_N_POSITIVE)
    parser.add_argument("--n-negative", type=int, default=DEFAULT_N_NEGATIVE)
    parser.add_argument("--positive-weight", type=float, default=DEFAULT_POSITIVE_WEIGHT)
    parser.add_argument("--lr", type=float, default=None, help="MLP learning rate override")
    parser.add_argument("--lm-lr", type=float, default=None, help="LM component learning rate override")
    parser.add_argument("--mlp-lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--max-target-len", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-pairs", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None, help="cap training pairs (debug)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _resolve_lrs(config: Mapping[str, Any], args: argparse.Namespace, device: str) -> Tuple[float, float, float]:
    lm_lr = args.lm_lr if args.lm_lr is not None else args.lr
    mlp_lr = args.mlp_lr if args.mlp_lr is not None else args.lr
    if lm_lr is None:
        lm_lr = cfg_get(config, "encoder_h", "lm_lr", default=1e-5)
    if mlp_lr is None:
        mlp_lr = cfg_get(config, "encoder_h", "mlp_lr", default=1e-4)
    return float(lm_lr), float(mlp_lr), float(args.positive_weight)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config) if args.config else {}
    model_key = args.model_key
    tuning = args.tuning if args.tuning not in (None, "none") else None
    device = args.device or cfg_get(config, "device", default="cpu")
    seed = int(args.seed if args.seed is not None else cfg_get(config, "seed", default=42))
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else cfg_get(config, "forecaster", "max_steps", default=DEFAULT_FORECASTER_STEPS)
    )
    batch_size = int(args.batch_size or cfg_get(config, "forecaster", "batch_size", default=DEFAULT_BATCH_SIZE))
    n_pos = int(args.n_positive or cfg_get(config, "forecaster", "n_positive", default=DEFAULT_N_POSITIVE))
    n_neg = int(args.n_negative or cfg_get(config, "forecaster", "n_negative", default=DEFAULT_N_NEGATIVE))
    lm_lr, mlp_lr, pos_w = _resolve_lrs(config, args, device)

    root = args.out_dir or artifact_root(config, model_key, tuning)
    gt_dir = args.gt_dir or os.path.join(root, DEFAULT_GT_DIRNAME)
    os.makedirs(root, exist_ok=True)

    checkpoint = None
    if not args.no_init_checkpoint:
        default_name = REPRESENTATION_FILENAME if args.method == "representation" else LOGIT_FILENAME
        for cand in (
            args.checkpoint,
            os.path.join(root, default_name),
            os.path.join(root, args.method, default_name),
        ):
            if cand and os.path.exists(cand):
                checkpoint = cand
                break

    logger.info("method=%s model=%s tuning=%s device=%s steps=%d batch=%d (%d+%d)",
                args.method, model_key, tuning, device, max_steps, batch_size, n_pos, n_neg)

    pairs = load_pairs(gt_dir)
    if args.max_pairs and len(pairs) > int(args.max_pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, int(args.max_pairs))
    test_pairs = load_pairs(gt_dir, prefer=("pairs_test.jsonl",))
    online = load_online_examples(gt_dir, model_key, tuning, config)
    upstream = load_upstream_examples(gt_dir, model_key, tuning, config)
    prior = None if args.no_prior else load_prior_for(gt_dir, config, model_key, tuning)

    if not pairs:
        raise SystemExit("no ground-truth pair records found in %s; run scripts/generate_ground_truth.py first" % gt_dir)
    if not upstream:
        logger.warning("no upstream examples found; pairs will be skipped")

    sampler = PairSampler(pairs, n_positive=n_pos, n_negative=n_neg, seed=seed)
    logger.info("train pairs=%d (pos=%d, neg=%d, prevalence=%.4f)", len(sampler), len(sampler.positives), len(sampler.negatives), sampler.positive_rate)

    forecaster = build_forecaster(
        method=args.method,
        model_key=model_key,
        config=config,
        device=device,
        dtype=args.dtype,
        fixed=bool(args.fixed),
        use_prior=not args.no_prior,
        prior=prior,
        positive_weight=pos_w,
        topk=args.topk,
        checkpoint=checkpoint,
    )
    online_by_index = build_index(online)

    out_name = REPRESENTATION_FILENAME if args.method == "representation" else LOGIT_FILENAME
    if args.method == "representation" and args.no_prior:
        out_name = "representation_forecaster_no_prior.pt"
    save_path = None if args.no_save else os.path.join(root, out_name)
    log_path = os.path.join(root, "train_log_%s.jsonl" % (args.method + ("_no_prior" if (args.method == "representation" and args.no_prior) else "")))

    eval_fn = make_eval_fn(args.method, test_pairs, online_by_index, upstream, prior, device, max_pairs=args.eval_pairs, max_target_len=args.max_target_len)

    start = time.time()
    if args.method == "representation":
        result = train_representation(
            forecaster=forecaster,
            sampler=sampler,
            online_by_index=online_by_index,
            upstream=upstream,
            prior=prior,
            max_steps=max_steps,
            batch_size=batch_size,
            lr_lm=lm_lr,
            lr_mlp=mlp_lr,
            positive_weight=pos_w,
            device=device,
            seed=seed,
            log_every=args.log_every,
            save_every=args.save_every,
            save_path=save_path,
            log_path=log_path,
            eval_fn=eval_fn,
            eval_every=args.eval_every,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        result = train_logit(
            forecaster=forecaster,
            sampler=sampler,
            online_by_index=online_by_index,
            upstream=upstream,
            max_steps=max_steps,
            batch_size=batch_size,
            lr_lm=lm_lr,
            lr_mlp=mlp_lr,
            positive_weight=pos_w,
            device=device,
            seed=seed,
            topk=args.topk,
            log_every=args.log_every,
            save_every=args.save_every,
            save_path=save_path,
            log_path=log_path,
            eval_fn=eval_fn,
            eval_every=args.eval_every,
            max_grad_norm=args.max_grad_norm,
            margin=args.margin,
            max_target_len=args.max_target_len,
        )

    final_metrics: Dict[str, Any] = {}
    if eval_fn is not None:
        try:
            final_metrics = eval_fn(forecaster)
        except Exception as exc:  # pragma: no cover
            logger.warning("final evaluation failed: %s", exc)

    summary = {
        "method": args.method,
        "fixed": bool(args.fixed),
        "use_prior": not bool(args.no_prior),
        "model_key": model_key,
        "tuning": tuning,
        "device": device,
        "seed": seed,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_weight": pos_w,
        "lr_lm": lm_lr,
        "lr_mlp": mlp_lr,
        "gt_dir": gt_dir,
        "train_pairs": len(sampler),
        "train_prevalence": sampler.positive_rate,
        "test_pairs": len(test_pairs),
        "n_online": len(online),
        "n_upstream": len(upstream),
        "checkpoint": save_path,
        "log_path": log_path,
        "elapsed_sec": time.time() - start,
        "final_metrics": final_metrics,
        "training": {k: v for k, v in result.items() if k != "history"},
    }
    summary_path = os.path.join(root, "train_summary_%s.json" % (args.method + ("_no_prior" if (args.method == "representation" and args.no_prior) else "")))
    write_json(summary_path, summary)
    logger.info("wrote training summary to %s", summary_path)
    logger.info("final metrics: %s", final_metrics)
    return summary


# --------------------------------------------------------------------------------------
# self test (offline, no model downloads)
# --------------------------------------------------------------------------------------
class _DummyEncoder:
    """Deterministic stand-in for ``EncoderH`` (self-test only)."""

    def __init__(self, dim: int = 16, vocab: int = 97):
        import torch
        import torch.nn as nn

        self.dim = int(dim)
        self.vocab = int(vocab)
        self.emb = nn.Embedding(self.vocab, self.dim)

    def _ids(self, texts: Sequence[str]) -> Any:
        import torch

        rows = []
        for text in texts:
            text = text or ""
            ids = [1 + (ord(c) % (self.vocab - 2)) for c in text[:16]] or [1]
            rows.append(ids)
        max_len = max(len(r) for r in rows)
        padded = [r + [0] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, dtype=torch.long)

    def encode(self, inputs, targets=None, mean_pool=True, batch_size=None, **kwargs):
        import torch

        texts = list(inputs)
        if targets is not None:
            texts = [str(t or "") + " " + str(g or "") for t, g in zip(texts, targets)]
        hidden = self.emb(self._ids(texts))
        if mean_pool:
            return hidden.mean(dim=1)
        return hidden

    def parameters(self):
        return self.emb.parameters()

    def trainable_parameters(self):
        return list(self.emb.parameters())

    def param_groups(self):
        return [{"params": list(self.emb.parameters()), "lr": 1e-4}]

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self


def _self_test() -> int:
    """Run a tiny end-to-end training loop on synthetic pairs."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print("torch unavailable: %s" % exc)
        return 0

    torch.manual_seed(0)
    n_upstream, n_online = 24, 6
    upstream = [{"index": j, "input": "upstream %d" % j, "target": "answer %d" % j} for j in range(n_upstream)]
    online = [{"index": i, "input": "online %d" % i, "target": "fix %d" % i} for i in range(n_online)]

    pairs: List[Dict[str, Any]] = []
    for i in range(n_online):
        for j in range(n_upstream):
            pairs.append({"i": i, "j": j, "z": 1 if (i + j) % 3 == 0 else 0})
    train_pairs = [p for p in pairs if p["i"] < 4]
    test_pairs = [p for p in pairs if p["i"] >= 4]

    sampler = PairSampler(train_pairs, n_positive=8, n_negative=8, seed=0)

    # ---------------- representation ----------------
    from src.forecasters.representation_based import RepresentationBasedForecaster

    encoder = _DummyEncoder(dim=8)
    model = RepresentationBasedForecaster(encoder=encoder, dim=8, use_prior=True, prior=None)
    result = train_representation(
        forecaster=model,
        sampler=sampler,
        online_by_index=build_index(online),
        upstream=upstream,
        prior=None,
        max_steps=12,
        batch_size=16,
        lr_lm=1e-4,
        lr_mlp=1e-4,
        positive_weight=0.1,
        device="cpu",
        seed=0,
        log_every=0,
        save_every=0,
        save_path=None,
        log_path=None,
        eval_fn=None,
        eval_every=0,
    )
    assert math.isfinite(result["final_loss"]), "representation loss is not finite"
    print("[self-test] representation OK (final loss %.4f)" % result["final_loss"])

    # ---------------- logit ----------------
    from src.forecasters.logit_based import LogitChangeTransferForecaster

    def _fake_topk(seed: int):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, 40, (3, 5), generator=g).tolist()
        val = torch.randn(3, 5, generator=g).tolist()
        return {"indices": idx, "values": val}

    logit_pairs = []
    for k, p in enumerate(train_pairs):
        rec = dict(p)
        rec["f0_i_token_logits"] = _fake_topk(k)
        rec["fi_i_token_logits"] = _fake_topk(k + 100)
        rec["f0_j_token_logits"] = _fake_topk(k + 200)
        rec["j_target"] = "answer %d" % p["j"]
        rec["i_target"] = "fix %d" % p["i"]
        logit_pairs.append(rec)
    logit_sampler = PairSampler(logit_pairs, n_positive=8, n_negative=8, seed=0)

    class _NoTok:
        pad_token_id = 0
        eos_token_id = None
        bos_token_id = None
        decoder_start_token_id = None

        def __call__(self, texts, **kwargs):
            ids = [1 + (ord(c) % 30) for c in (texts[0] if isinstance(texts, list) else str(texts))[:4]] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    logit_encoder = _DummyEncoder(dim=8)
    logit_encoder.tokenizer = _NoTok()
    logit_model = LogitChangeTransferForecaster(encoder=logit_encoder, dim=8, topk=100)
    logit_result = train_logit(
        forecaster=logit_model,
        sampler=logit_sampler,
        online_by_index=build_index(online),
        upstream=upstream,
        max_steps=8,
        batch_size=16,
        lr_lm=1e-4,
        lr_mlp=1e-4,
        positive_weight=0.1,
        device="cpu",
        seed=0,
        topk=100,
        log_every=0,
        save_every=0,
        save_path=None,
        log_path=None,
        eval_fn=None,
        eval_every=0,
    )
    print("[self-test] logit OK (final loss %s, skipped=%d)" % (logit_result["final_loss"], logit_result["skipped_pairs"]))

    # ---------------- helpers ----------------
    assert candidates_from([[[1, 2]], [[3, 4]]], target_ids=None) == [1, 2, 3, 4]
    dense = densify_topk([[1, 2]], [[0.5, 0.25]], [1, 2, 3])
    assert dense.shape == (1, 3) and abs(float(dense[0, 0]) - 0.5) < 1e-6
    d = delta_matrix([[1]], [[1.0]], [[1]], [[3.0]], [1, 2])
    assert abs(float(d[0, 0]) - 2.0) < 1e-6
    assert binary_f1([1, 0, 1], [1, 1, 0]) > 0.0
    print("[self-test] helpers OK")
    print("[self-test] PASSED")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "training"}, indent=2, default=_default_serializer))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
