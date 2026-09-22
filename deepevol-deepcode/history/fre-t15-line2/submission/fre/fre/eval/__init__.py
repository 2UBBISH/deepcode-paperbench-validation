"""Evaluation utilities for FRE (Zero-Shot RL via Functional Reward Encodings).

This package groups the zero-shot evaluation harness
(:mod:`fre.eval.zero_shot_eval`) and the prior-method baselines
(:mod:`fre.eval.baselines`) used in Section 5.2 of the paper:

* ``FRE``               -- the method itself (evaluated via ``zero_shot_eval``).
* ``GC-BC`` / ``GC-IQL`` -- goal-conditioned behavioural cloning / implicit
  Q-learning baselines trained inside the same codebase.
* ``OPAL``              -- privileged-evaluation baseline re-using the FRE
  transformer encoder but with 10 Gaussian skills (best rollout taken).

Note (paper): FB/SF are trained with the separate
``github.com/facebookresearch/controllable_agent`` codebase and are therefore
not implemented here (see the addendum); only a thin adapter/constants live in
this package.

The symbols are resolved lazily (PEP 562) so that ``import fre.eval`` stays
cheap and never hard-requires torch / MuJoCo / D4RL just to be importable.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = [
    # zero-shot harness
    "ZeroShotEvaluator",
    "evaluate_zero_shot",
    "evaluate_registered_suites",
    "aggregate_seeds",
    "format_seed_summary",
    "ContextSet",
    "EpisodeResult",
    "TaskResult",
    "SuiteResult",
    "EvalReport",
    "sample_task_context",
    "encode_context",
    "resolve_normalization",
    "normalize_return",
    "policy_action",
    # protocol constants (Section 5.2 / Table 1)
    "NUM_EVAL_EPISODES",
    "NUM_TRAINING_SEEDS",
    "FRE_CONTEXT_SAMPLES",
    "FB_SF_CONTEXT_SAMPLES",
    "NORMALIZED_RETURN_MIN",
    "NORMALIZED_RETURN_MAX",
    # baselines
    "GCBCAgent",
    "GCIQLAgent",
    "OpalAgent",
    "BaselineResult",
    "make_gc_bc_agent",
    "make_gc_iql_agent",
    "make_opal_agent",
    "evaluate_baseline",
    "GC_BC_DEFAULTS",
    "GC_IQL_DEFAULTS",
    "OPAL_DEFAULTS",
]

# public name -> defining submodule
_LAZY_ATTRS: Dict[str, str] = {
    # fre.eval.zero_shot_eval
    "ZeroShotEvaluator": "zero_shot_eval",
    "evaluate_zero_shot": "zero_shot_eval",
    "evaluate_registered_suites": "zero_shot_eval",
    "aggregate_seeds": "zero_shot_eval",
    "format_seed_summary": "zero_shot_eval",
    "ContextSet": "zero_shot_eval",
    "EpisodeResult": "zero_shot_eval",
    "TaskResult": "zero_shot_eval",
    "SuiteResult": "zero_shot_eval",
    "EvalReport": "zero_shot_eval",
    "sample_task_context": "zero_shot_eval",
    "encode_context": "zero_shot_eval",
    "resolve_normalization": "zero_shot_eval",
    "normalize_return": "zero_shot_eval",
    "policy_action": "zero_shot_eval",
    "NUM_EVAL_EPISODES": "zero_shot_eval",
    "NUM_TRAINING_SEEDS": "zero_shot_eval",
    "FRE_CONTEXT_SAMPLES": "zero_shot_eval",
    "FB_SF_CONTEXT_SAMPLES": "zero_shot_eval",
    "NORMALIZED_RETURN_MIN": "zero_shot_eval",
    "NORMALIZED_RETURN_MAX": "zero_shot_eval",
    # fre.eval.baselines.gc_bc
    "GCBCAgent": "baselines.gc_bc",
    "make_gc_bc_agent": "baselines.gc_bc",
    "GC_BC_DEFAULTS": "baselines.gc_bc",
    # fre.eval.baselines.gc_iql
    "GCIQLAgent": "baselines.gc_iql",
    "make_gc_iql_agent": "baselines.gc_iql",
    "GC_IQL_DEFAULTS": "baselines.gc_iql",
    # fre.eval.baselines.opal
    "OpalAgent": "baselines.opal",
    "make_opal_agent": "baselines.opal",
    "OPAL_DEFAULTS": "baselines.opal",
    # fre.eval.baselines (shared helpers)
    "BaselineResult": "baselines",
    "evaluate_baseline": "baselines",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy loader
    """PEP 562 lazy resolution of the public ``fre.eval`` surface."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # memoize
    return value


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(set(globals()) | set(__all__))
