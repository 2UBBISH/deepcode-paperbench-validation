"""Online adaptation package for BBox-Adapter (Section 3.4, Algorithm 1).

This package owns:

* :mod:`bbox_adapter.training.buffers` -- per-query positive/negative sample
  stores, the positive-selection functions ``SEL(.)`` (ground truth, GPT-4 AI
  feedback, combined) implementing Eq. (5) / Eq. (6), and outcome supervision.
* :mod:`bbox_adapter.training.online_adaptation` -- Algorithm 1: the outer
  ``t = 1 .. T`` loop that refreshes the contrastive sets from the current
  adapted inference ``p_{theta_t}`` and updates the energy adapter
  ``theta_{t+1} = theta_t - eta * grad`` (Eq. 7) with AdamW
  (``eta = 5e-6``, weight decay ``0.01``, batch size ``64``, 6000 steps).

Nothing in this package ever requests logprobs, hidden states, or gradients of
the black-box LLM; only the small adapter ``g_theta`` carries a gradient.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbox_adapter.training")

__all__: List[str] = []


def _extend(names: Optional[List[str]]) -> None:
    """Append ``names`` to ``__all__`` without duplicating entries."""
    if not names:
        return
    for name in names:
        if name not in __all__:
            __all__.append(name)


# --------------------------------------------------------------------------- #
# Hyper-parameter constants (Appendix H.2 / plan defaults)
# --------------------------------------------------------------------------- #
LEARNING_RATE = 5e-6
BATCH_SIZE = 64
TRAINING_STEPS = 6000
WEIGHT_DECAY = 0.01
BEAM_SIZE = 3
MAX_LEN = 512
TEMPERATURE = 1.0

_ADAM_BETAS = (0.9, 0.999)
_WARMUP_FRACTION = 0.1
_DEFAULT_T = 4
_DEFAULT_M = 5
_DEFAULT_K = 5
_DEFAULT_ALPHA = 1e-2

# --------------------------------------------------------------------------- #
# Sel mode constants (mirrored from training.buffers)
# --------------------------------------------------------------------------- #
SEL_MODE_GROUND_TRUTH = "ground_truth"
SEL_MODE_AI_FEEDBACK = "ai_feedback"
SEL_MODE_COMBINED = "combined"
SEL_MODE_RANDOM = "random"
SEL_MODES = (
    SEL_MODE_GROUND_TRUTH,
    SEL_MODE_AI_FEEDBACK,
    SEL_MODE_COMBINED,
    SEL_MODE_RANDOM,
)

# --------------------------------------------------------------------------- #
# buffers.py -- Eq. (5) / Eq. (6), SEL, outcome supervision
# --------------------------------------------------------------------------- #
_BUFFERS_AVAILABLE = False
try:  # pragma: no cover - defensive import
    from .buffers import (  # noqa: F401
        SEL_MODE_ALIASES,
        SEL_MODE_AI_FEEDBACK as _BUF_SEL_AI,
        SEL_MODE_COMBINED as _BUF_SEL_COMBINED,
        SEL_MODE_GROUND_TRUTH as _BUF_SEL_GT,
        SEL_MODE_RANDOM as _BUF_SEL_RANDOM,
        BufferConfig,
        QuerySamples,
        SampleBuffer,
        ai_feedback_select,
        apply_outcome_supervision,
        candidate_key,
        combined_select,
        deduplicate,
        ground_truth_select,
        initial_query_samples,
        initialize_buffer,
        make_selector,
        random_select,
        update_negatives,
        update_positive,
        update_query_samples,
    )

    _BUFFERS_AVAILABLE = True
    _extend(
        [
            "BufferConfig",
            "QuerySamples",
            "SampleBuffer",
            "ai_feedback_select",
            "apply_outcome_supervision",
            "candidate_key",
            "combined_select",
            "deduplicate",
            "ground_truth_select",
            "initial_query_samples",
            "initialize_buffer",
            "make_selector",
            "random_select",
            "update_negatives",
            "update_positive",
            "update_query_samples",
            "SEL_MODE_ALIASES",
        ]
    )
except Exception as exc:  # pragma: no cover
    logger.debug("training.buffers unavailable: %s", exc)

# --------------------------------------------------------------------------- #
# online_adaptation.py -- Algorithm 1
# --------------------------------------------------------------------------- #
_ONLINE_AVAILABLE = False
try:  # pragma: no cover - defensive import
    from .online_adaptation import (  # noqa: F401
        ContrastiveSetDataset,
        OnlineAdaptationConfig,
        OnlineAdapter,
        collate_contrastive,
        run_online_adaptation,
        train_adapter,
        uniform_temperature,
    )

    _ONLINE_AVAILABLE = True
    _extend(
        [
            "ContrastiveSetDataset",
            "OnlineAdaptationConfig",
            "OnlineAdapter",
            "collate_contrastive",
            "run_online_adaptation",
            "train_adapter",
            "uniform_temperature",
        ]
    )
except Exception as exc:  # pragma: no cover
    logger.debug("training.online_adaptation unavailable: %s", exc)


# --------------------------------------------------------------------------- #
# Convenience factories
# --------------------------------------------------------------------------- #
def get_config(
    dataset: Optional[str] = None,
    size: Optional[str] = None,
    *,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Optional["OnlineAdaptationConfig"]:  # noqa: F821
    """Build an :class:`OnlineAdaptationConfig` for ``dataset`` / ``size``.

    Paper defaults (Appendix H.2) are applied for every unspecified knob:
    ``lr = 5e-6``, ``batch_size = 64``, ``max_train_steps = 6000``,
    ``weight_decay = 0.01``, ``beam_size = 3``, ``max_len = 512``,
    ``temperature = 1.0``.
    """
    if not _ONLINE_AVAILABLE:
        raise ImportError(
            "bbox_adapter.training.online_adaptation is required for get_config"
        )

    payload: Dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "to_dict"):
            payload.update(config.to_dict())
        elif isinstance(config, dict):
            payload.update(config)
        else:  # duck-typed namespace
            payload.update(vars(config))

    payload.setdefault("lr", LEARNING_RATE)
    payload.setdefault("batch_size", BATCH_SIZE)
    payload.setdefault("max_train_steps", TRAINING_STEPS)
    payload.setdefault("weight_decay", WEIGHT_DECAY)
    payload.setdefault("betas", _ADAM_BETAS)
    payload.setdefault("warmup_steps", 0)
    payload.setdefault("n_iterations", _DEFAULT_T)
    payload.setdefault("n_candidates", _DEFAULT_M)
    payload.setdefault("k_init", _DEFAULT_K)
    payload.setdefault("beam_size", BEAM_SIZE)
    payload.setdefault("max_length", MAX_LEN)
    payload.setdefault("max_len", MAX_LEN)
    payload.setdefault("temperature", TEMPERATURE)
    payload.setdefault("alpha", _DEFAULT_ALPHA)
    if dataset is not None:
        payload["dataset"] = dataset
    if size is not None:
        payload["size"] = size
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return OnlineAdaptationConfig.from_dict(payload)


def get_sampler(
    adapter: Any,
    generator: Any = None,
    *,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Optional["OnlineAdapter"]:  # noqa: F821
    """Build an :class:`OnlineAdapter` (Algorithm 1 driver) if available."""
    if not _ONLINE_AVAILABLE:
        raise ImportError(
            "bbox_adapter.training.online_adaptation is required for get_sampler"
        )
    return OnlineAdapter(adapter, generator, config=config, **kwargs)


def get_buffer(config: Optional[Any] = None, **kwargs: Any) -> Optional["SampleBuffer"]:  # noqa: F821
    """Build a :class:`SampleBuffer` from a config mapping / dataclass."""
    if not _BUFFERS_AVAILABLE:
        raise ImportError("bbox_adapter.training.buffers is required for get_buffer")
    if isinstance(config, BufferConfig):
        return SampleBuffer(config)
    payload: Dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "to_dict"):
            payload.update(config.to_dict())
        elif isinstance(config, dict):
            payload.update(config)
        else:  # duck-typed namespace
            payload.update(vars(config))
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return SampleBuffer(BufferConfig.from_dict(payload) if payload else None)


def get_selector_for(
    mode: str = SEL_MODE_GROUND_TRUTH,
    *,
    config: Optional[Any] = None,
    rater: Any = None,
    **kwargs: Any,
):
    """Build the ``SEL(.)`` callable for a positive-sample setting.

    ``ground_truth`` / ``combined`` / ``random`` are owned by
    :mod:`bbox_adapter.training.buffers`; ``ai_feedback`` additionally accepts a
    GPT-4 rater (see :mod:`bbox_adapter.feedback.ai_feedback`).
    """
    if not _BUFFERS_AVAILABLE:
        raise ImportError("bbox_adapter.training.buffers is required")
    return make_selector(mode, config=config, rater=rater, **kwargs)


def describe() -> Dict[str, Any]:
    """Return metadata about the training package (used by run scripts)."""
    return {
        "module": "bbox_adapter.training",
        "paper": "Lightweight Adapting for Black-Box Large Language Models",
        "sections": {
            "buffers": "Section 3.4 (Eq. 5, Eq. 6), Appendix G",
            "online_adaptation": "Section 3.4 (Algorithm 1), Section 3.2 (Eq. 7)",
        },
        "available": {
            "buffers": _BUFFERS_AVAILABLE,
            "online_adaptation": _ONLINE_AVAILABLE,
        },
        "defaults": {
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_train_steps": TRAINING_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(_ADAM_BETAS),
            "warmup_fraction": _WARMUP_FRACTION,
            "beam_size": BEAM_SIZE,
            "max_len": MAX_LEN,
            "temperature": TEMPERATURE,
            "n_iterations_T": _DEFAULT_T,
            "n_candidates_M": _DEFAULT_M,
            "k_init_K": _DEFAULT_K,
            "alpha": _DEFAULT_ALPHA,
        },
        "sel_modes": list(SEL_MODES),
        "blackbox_contract": (
            "text-only: no logprobs / hidden states / gradients of the black-box LLM"
        ),
    }


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test of the training package surface."""
    checks: Dict[str, Any] = {}
    checks["buffers_available"] = _BUFFERS_AVAILABLE
    checks["online_available"] = _ONLINE_AVAILABLE
    checks["paper_defaults"] = (
        LEARNING_RATE == 5e-6
        and BATCH_SIZE == 64
        and TRAINING_STEPS == 6000
        and WEIGHT_DECAY == 0.01
        and BEAM_SIZE == 3
        and MAX_LEN == 512
        and TEMPERATURE == 1.0
    )
    checks["sel_modes"] = set(SEL_MODES) >= {
        "ground_truth",
        "ai_feedback",
        "combined",
    }
    checks["exports"] = sorted(set(__all__))
    checks["describe"] = describe()
    return checks


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(_self_test(), indent=2, default=str))
