"""Training subpackage for Functional Reward Encodings (FRE).

This package bundles the three training components of the FRE paper
("Zero-Shot Reinforcement Learning via Functional Reward Encodings"):

* :mod:`fre.training.fre_trainer` -- Algorithm 1, phase 1: maximize Equation (6),
  the variational lower bound of the information bottleneck
  ``I(L_eta^d ; Z) - beta * I(L_eta^e ; Z)``, optimizing the permutation-invariant
  transformer encoder ``p_theta(z | .)`` and the feed-forward decoder
  ``q_theta(eta(s^d) | s^d, z)``.  Adam, lr 1e-4, batch 512, K=32, K'=8, beta=0.01.
* :mod:`fre.training.iql` -- the IQL losses (expectile 0.8, AWR temperature 3.0,
  target update rate 0.001, discount 0.88) on the z-conditioned Q/V/policy nets.
* :mod:`fre.training.strided` -- the "strided" two-phase controller implementing
  Algorithm 1 end to end: train the encoder with decoder gradients only, freeze it
  once the encoder loss converges, then train the RL networks on the frozen latents.

The imports are performed lazily (PEP 562) so that ``import fre.training`` always
succeeds even while individual modules are still being built out, and so that the
heavy ``torch`` dependency is only paid for on first attribute access.
"""

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------------------
__all__: List[str] = [
    # fre.training.fre_trainer -- Algorithm 1, phase 1 (Equation 6)
    "FRETrainer",
    "FRETrainerConfig",
    "FRETrainOutput",
    "make_fre_trainer",
    "train_fre_step",
    "fre_reconstruction_kl_loss",
    # fre.training.iql -- IQL losses / update engine (phase 2)
    "IQLTrainer",
    "IQLConfig",
    "IQLLosses",
    "IQLLoss",
    "expectile_loss",
    "awr_weights",
    "target_q_from_value",
    "soft_update",
    "make_iql_trainer",
    "train_iql_step",
    # fre.training.strided -- strided controller (Algorithm 1, both phases)
    "StridedTrainer",
    "StridedConfig",
    "StridedTrainerResult",
    "ConvergenceDetector",
    "freeze_encoder",
    "unfreeze_encoder",
    "encoder_is_frozen",
    "assert_encoder_frozen",
    "train_fre_strided",
]

# name -> submodule that defines it (all resolved lazily in __getattr__)
_LAZY_ATTRS: Dict[str, str] = {
    # fre.training.fre_trainer
    "FRETrainer": "fre_trainer",
    "FRETrainerConfig": "fre_trainer",
    "FRETrainOutput": "fre_trainer",
    "make_fre_trainer": "fre_trainer",
    "train_fre_step": "fre_trainer",
    "fre_reconstruction_kl_loss": "fre_trainer",
    # fre.training.iql
    "IQLTrainer": "iql",
    "IQLConfig": "iql",
    "IQLLosses": "iql",
    "IQLLoss": "iql",
    "expectile_loss": "iql",
    "awr_weights": "iql",
    "target_q_from_value": "iql",
    "soft_update": "iql",
    "make_iql_trainer": "iql",
    "train_iql_step": "iql",
    # fre.training.strided
    "StridedTrainer": "strided",
    "StridedConfig": "strided",
    "StridedTrainerResult": "strided",
    "ConvergenceDetector": "strided",
    "freeze_encoder": "strided",
    "unfreeze_encoder": "strided",
    "encoder_is_frozen": "strided",
    "assert_encoder_frozen": "strided",
    "train_fre_strided": "strided",
}

# Table 3 hyper-parameters, mirrored here for convenience of config code that only
# wants the numbers without importing torch.
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BETA = 0.01
DEFAULT_EXPECTILE = 0.8
DEFAULT_AWR_TEMPERATURE = 3.0
DEFAULT_DISCOUNT = 0.88
DEFAULT_TAU = 0.001
DEFAULT_NUM_ENCODER_STATES = 32  # K
DEFAULT_NUM_DECODER_STATES = 8  # K'
DEFAULT_ENCODER_TRAINING_STEPS = 150_000
DEFAULT_POLICY_TRAINING_STEPS = 850_000
LONG_ENCODER_TRAINING_STEPS = 1_000_000  # ExORL / Kitchen
LONG_POLICY_TRAINING_STEPS = 1_000_000  # ExORL / Kitchen

#: (encoder steps, policy steps) budgets per domain, from Table 3's footnotes.
DOMAIN_STEP_BUDGETS: Dict[str, tuple] = {
    "antmaze": (DEFAULT_ENCODER_TRAINING_STEPS, DEFAULT_POLICY_TRAINING_STEPS),
    "exorl": (LONG_ENCODER_TRAINING_STEPS, LONG_POLICY_TRAINING_STEPS),
    "kitchen": (LONG_ENCODER_TRAINING_STEPS, LONG_POLICY_TRAINING_STEPS),
}

DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "beta": DEFAULT_BETA,
    "expectile": DEFAULT_EXPECTILE,
    "awr_temperature": DEFAULT_AWR_TEMPERATURE,
    "discount": DEFAULT_DISCOUNT,
    "tau": DEFAULT_TAU,
    "num_encoder_states": DEFAULT_NUM_ENCODER_STATES,
    "num_decoder_states": DEFAULT_NUM_DECODER_STATES,
    "encoder_training_steps": DEFAULT_ENCODER_TRAINING_STEPS,
    "policy_training_steps": DEFAULT_POLICY_TRAINING_STEPS,
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for the re-exported training symbols."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - developer error
        raise AttributeError(
            f"module {__name__}.{module_name!r} does not define {name!r}"
        ) from exc
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
