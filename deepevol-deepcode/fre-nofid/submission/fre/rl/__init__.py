"""Reinforcement-learning subpackage for FRE.

Aggregates the z-conditioned Implicit Q-Learning (IQL) trainer and the offline
replay buffer used by both training phases of

    "Zero-Shot Reinforcement Learning via Functional Reward Encodings" (FRE).

The imports are defensive so that a partial install (e.g. missing ``torch`` or
``numpy``) still loads whichever components are available.
"""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = []


def _try_import(module: str, names: List[str]) -> None:
    """Best-effort re-export of ``names`` from ``module``.

    Failures are swallowed so that a missing optional dependency in one
    submodule does not prevent the rest of the package from importing.
    """
    try:
        mod = __import__(module, fromlist=names)
    except Exception:  # pragma: no cover - defensive
        return
    for name in names:
        try:
            value = getattr(mod, name)
        except AttributeError:
            continue
        globals()[name] = value
        if name not in __all__:
            __all__append(name)


def __all__append(name: str) -> None:
    __all__.append(name)


# ---------------------------------------------------------------------------
# Replay buffer (offline transitions, dataset -> canonical arrays, HER support)
# ---------------------------------------------------------------------------
_try_import(
    "fre.rl.replay_buffer",
    [
        # core containers
        "Batch",
        "ReplayBuffer",
        "OfflineReplayBuffer",
        "TrajectoryReplayBuffer",
        # helpers
        "make_batch",
        "d4rl_dict_to_arrays",
        "concatenate_buffers",
        "batch_to_device",
        "add_reward_to_batch",
        "dataset_observation_stats",
        # constants
        "TRANSITION_KEYS",
        "DEFAULT_CAPACITY",
    ],
)

# ---------------------------------------------------------------------------
# z-conditioned IQL (phase 2 of Algorithm 1, Sec 4.3)
# ---------------------------------------------------------------------------
_try_import(
    "fre.rl.iql",
    [
        # trainer + config
        "IQLConfig",
        "IQLTrainer",
        "IQL",
        "IQLAgent",
        # losses
        "expectile_loss",
        "asymmetric_l2_loss",
        "awr_weights",
        # hyperparameters
        "DEFAULT_DISCOUNT",
        "DEFAULT_EXPECTILE",
        "DEFAULT_AWR_TEMPERATURE",
        "DEFAULT_AWR_MAX_WEIGHT",
        "DEFAULT_TARGET_UPDATE_RATE",
        "DEFAULT_LR",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_MAX_GRAD_NORM",
        "DEFAULT_NUM_CANDIDATE_ACTIONS",
    ],
)

# Fallback: expose a stable alias even if the submodule could not be imported.
if "IQLTrainer" in globals() and "IQL" not in globals():  # pragma: no cover
    globals()["IQL"] = globals()["IQLTrainer"]
    __all__append("IQL")


def __getattr__(name: str) -> Any:  # pragma: no cover - defensive helper
    raise AttributeError(
        f"module 'fre.rl' has no attribute {name!r}. "
        f"Available names: {sorted(__all__)}"
    )
