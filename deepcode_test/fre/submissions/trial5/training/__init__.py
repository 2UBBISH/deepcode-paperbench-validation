"""
Training module for Functional Reward Encodings (FRE).

Contains:
- train_encoder: Phase 1 training of the FRE encoder/decoder
- train_rl: Phase 2 training of the IQL agent with frozen encoder
- trainer: Main orchestrator for strided training
"""

from .train_encoder import FREEncoderTrainer
from .train_rl import IQLTrainer
from .trainer import FRETrainer

__all__ = [
    "FREEncoderTrainer",
    "IQLTrainer",
    "FRETrainer",
]