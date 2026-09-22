"""The energy based adapter ``g_theta`` and the ranking based NCE loss."""

from .base import BaseAdapter, TrainingExample, TrainLog
from .energy import EnergyAdapter, PAIR_TEMPLATE, format_pairs
from .losses import ranking_nce_loss, ranking_nce_softmax_loss, l2_energy_regularizer
from .trainer import AdapterTrainer

__all__ = [
    "BaseAdapter",
    "TrainingExample",
    "TrainLog",
    "EnergyAdapter",
    "PAIR_TEMPLATE",
    "format_pairs",
    "ranking_nce_loss",
    "ranking_nce_softmax_loss",
    "l2_energy_regularizer",
    "AdapterTrainer",
]
