"""Common interface for knowledge-retention methods."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class RetentionConfig:
    """Configuration shared by all knowledge-retention methods.

    Attributes
    ----------
    method:
        One of ``"none"``, ``"ewc"``, ``"bc"``, ``"ks"``, ``"em"``.
    coefficient:
        The multiplier applied to the auxiliary retention loss.  The paper uses
        2.0 for BC in NetHack, 0.5 for KS in NetHack (with exponential decay),
        2e6 for the EWC regularization coefficient in NetHack, 100 for the EWC
        actor regularization coefficient in Meta-World and 1 for BC in
        Meta-World (Table 3).
    decay:
        Exponential decay applied to ``coefficient`` every *training step*
        (NetHack uses 0.99998 for KS).  ``1.0`` means no decay.
    regularize_critic:
        The paper never regularizes the critic (Appendix C.5), so this defaults
        to ``False``.
    """

    method: str = "none"
    coefficient: float = 1.0
    decay: float = 1.0
    regularize_critic: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"none", "ewc", "bc", "ks", "em"}
        if self.method not in valid:
            raise ValueError(f"Unknown retention method {self.method!r}; expected one of {sorted(valid)}")


class RetentionMethod(abc.ABC):
    """Base class for knowledge-retention methods.

    A retention method exposes a single scalar ``aux_loss`` that is added to the
    reinforcement-learning objective::

        total_loss = rl_loss + coefficient * retention.aux_loss(...)

    and an optional ``on_train_step`` hook used for coefficient schedules.
    """

    def __init__(self, config: Optional[RetentionConfig] = None) -> None:
        self.config = config or RetentionConfig()
        self._step = 0

    # ------------------------------------------------------------------ API
    @abc.abstractmethod
    def aux_loss(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the (un-scaled) auxiliary retention loss."""

    def on_train_step(self) -> None:
        """Advance any internal schedules by one training step."""

        self._step += 1

    @property
    def coefficient(self) -> float:
        return self.config.coefficient

    @property
    def step(self) -> int:
        return self._step

    # Convenience -------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"step": self._step, "config": self.config.__dict__}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step = state.get("step", 0)

    @staticmethod
    def build(config: RetentionConfig):
        """Factory returning the concrete method described by ``config``."""

        from .distillation import BehavioralCloning, Kickstarting
        from .episodic_memory import EpisodicMemory
        from .ewc import EWC

        if config.method == "none":
            return None
        if config.method == "ewc":
            return EWC(config)
        if config.method == "bc":
            return BehavioralCloning(config)
        if config.method == "ks":
            return Kickstarting(config)
        if config.method == "em":
            return EpisodicMemory(config)
        raise ValueError(f"Unknown retention method {config.method!r}")
