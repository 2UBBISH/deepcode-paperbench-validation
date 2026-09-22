"""Per-policy rollout buffers for SAPG (data sets ``D_1 .. D_M``).

SAPG collects each block of environments with its own policy and therefore needs
one buffer *per policy* rather than a single monolithic rollout buffer.  This
package exposes:

* :class:`RolloutBuffer` -- the storage for one policy's dataset ``D_j``
  (observations, actions, rewards, log-probabilities, values, dones, ``phi_j``),
  together with advantage estimation (GAE), the 3-step on-policy critic targets
  of Eq. (5) and the 1-step off-policy critic targets of Eq. (7), plus the
  importance-correction term ``mu`` of Eq. (3).
* :class:`BufferSet` -- a thin container for ``D_1 .. D_M`` with 1-based
  leader/follower accessors (the leader is policy ``i = 1``, Section 4.3).
* :class:`OffPolicyBatch` -- the fused leader batch ``D'_1`` built from the
  union of the follower datasets, carrying a ``source_policy`` index per
  transition so the source ratio ``pi_j`` can be evaluated.
* :func:`build_off_policy_batch` -- builds ``D'_1`` with the uniform
  subsampling rule of Section 4.3 (``|D'_1| = |D_1|``).
* :func:`compute_n_step_targets`, :func:`compute_one_step_targets` and
  :func:`mu_from_logprobs` -- the small functional helpers behind the above.

Like the other sub-packages, the heavy module is imported lazily through
:pep:`562` module-level ``__getattr__`` so that ``import sapg.buffers`` remains
free of side effects and does not require ``torch`` immediately.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # storage
    "RolloutBuffer",
    "BufferSet",
    "OffPolicyBatch",
    # off-policy batch construction (Section 4.3)
    "build_off_policy_batch",
    "subsample_indices",
    # importance correction (Eq. 3)
    "mu_from_logprobs",
    # critic targets (Eqs. 5-6-7)
    "compute_n_step_targets",
    "compute_one_step_targets",
]

#: Maps every public name to the submodule that defines it.  A single source of
#: truth keeps ``__all__`` and the lazy dispatch below consistent.
_LAZY_EXPORTS: Dict[str, str] = {
    "RolloutBuffer": "rollout_buffer",
    "BufferSet": "rollout_buffer",
    "OffPolicyBatch": "rollout_buffer",
    "build_off_policy_batch": "rollout_buffer",
    "subsample_indices": "rollout_buffer",
    "mu_from_logprobs": "rollout_buffer",
    "compute_n_step_targets": "rollout_buffer",
    "compute_one_step_targets": "rollout_buffer",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve the public buffer API (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            f"available names: {sorted(__all__)}"
        )
    module = importlib.import_module(f".{module_name}", __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {__name__!r} could not resolve {name!r} from "
            f"{module.__name__!r}: {exc}"
        ) from exc


def __dir__() -> List[str]:
    """Introspection helper listing the public buffer API."""
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
