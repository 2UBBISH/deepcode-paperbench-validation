"""Trial configuration, constants, and reduced-scale defaults.

The paper's full experimental scale uses 3000 training iterations, 10 rounds,
20000 HPR samples, and pyloric round sizes of 30000/20000. The execution matrix
overrides these with the trial's :code:`parameters` values (including reduced
smoke values), so the defaults below are reference-scale fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


LEARNING_RATE: float = 1e-4
VALIDATION_FRACTION: float = 0.15
EARLY_STOPPING_PATIENCE: int = 1000
MAX_TRAINING_ITERATIONS: int = 3000
TIME_EMBEDDING_DIM: int = 64
HPR_EPSILON: float = 5e-4

DEFAULT_LEARNING_RATE = LEARNING_RATE
DEFAULT_VALIDATION_FRACTION = VALIDATION_FRACTION


@dataclass
class TrialConfig:
    """Configuration for one execution-matrix trial cell.

    Values are read from ``job/trial.json`` by :meth:`from_trial`. The fields
    mirror the keys in the freezing execution spec so later modules can use a
    single typed object instead of passing raw dictionaries.
    """

    method: str
    setting: str
    seed: int = 0
    simulation_budget: int = 1000
    training_steps: int = 3000
    num_rounds: int = 10
    posterior_samples: int = 128
    hpr_samples: int = 256
    batch_size: int = 50
    pyloric_initial_simulations: int = 30000
    pyloric_added_per_round: int = 20000
    device: str = "cpu"

    @classmethod
    def from_trial(cls, trial: dict) -> "TrialConfig":
        """Construct a TrialConfig from a trial dictionary.

        The frozen trial JSON stores parameter values as floats (for example
        ``100.0``); this method rounds them to integers for the fields that are
        naturally counts.
        """

        parameters = {p["key"]: p["value"] for p in trial.get("parameters", [])}

        def _int(key: str, default: int) -> int:
            value = parameters.get(key, default)
            if value is None:
                return default
            return int(round(float(value)))

        return cls(
            method=trial.get("method", ""),
            setting=trial.get("setting", ""),
            seed=int(trial.get("seed", 0)),
            simulation_budget=_int("simulation_budget", 1000),
            training_steps=_int("training_steps", 3000),
            num_rounds=_int("num_rounds", 10),
            posterior_samples=_int("posterior_samples", 128),
            hpr_samples=_int("hpr_samples", 256),
            batch_size=_int("batch_size", 50),
            pyloric_initial_simulations=_int("pyloric_initial_simulations", 30000),
            pyloric_added_per_round=_int("pyloric_added_per_round", 20000),
            device=str(trial.get("device", "cpu")),
        )


@dataclass
class SDEConfig:
    """Selects the forward noising SDE and stores its hyperparameters.

    For VE, ``sigma_min`` and ``sigma_max`` define the variance schedule. For
    VP, ``beta_min`` and ``beta_max`` define the linear beta schedule. Only the
    fields relevant to ``kind`` are consumed by :func:`impl.sde.build_sde`.
    """

    kind: str = "ve"
    sigma_min: float = 0.05
    sigma_max: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 11.0

    @classmethod
    def ve(cls, sigma_min: float = 0.05, sigma_max: float = 1.0) -> "SDEConfig":
        """Return a variance-exploding configuration."""
        return cls(kind="ve", sigma_min=sigma_min, sigma_max=sigma_max)

    @classmethod
    def vp(cls, beta_min: float = 0.1, beta_max: float = 11.0) -> "SDEConfig":
        """Return a variance-preserving configuration."""
        return cls(kind="vp", beta_min=beta_min, beta_max=beta_max)


def ve_sigma_min_for_setting(setting: str) -> float:
    """Return the paper's task-dependent VE ``sigma_min``.

    The paper sets ``sigma_min = 0.01`` for the two-dimensional SIR and Two
    Moons tasks and ``sigma_min = 0.05`` for all other tasks.
    """

    if setting in {"sir", "two_moons"}:
        return 0.01
    return 0.05
