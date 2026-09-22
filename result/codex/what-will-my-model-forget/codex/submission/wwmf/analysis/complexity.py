"""Sec. 5.3 -- computational efficiency of the forecasting methods (Table 5).

Notation of the paper: ``N_PT`` upstream pretraining examples, ``T`` maximal
input/output length, ``H`` representation dimension, ``V`` vocabulary size and
``Fw(N)`` the cost of running inference with ``N`` examples.  Logits and
representations of ``D_PT`` are cached, so they are computed once and reused for
every online learning example.

Table 5:

    Method / Setup     Head                   Full FT
    Threshold          O(N_PT)                O(N_PT)
    Trainable Logit    O(N_PT T^2 (H + V))    O(N_PT T^2 (H + V))
    Representation     O(N_PT H)              O(N_PT H)
    Ground Truth       O(N_PT T H V)          O(Fw(N))

Appendix C additionally reports FLOP counts; ``estimate_flops`` implements the
same accounting in closed form (see the README for the mapping).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

COMPLEXITY_TABLE: Dict[str, Dict[str, str]] = {
    "Threshold": {"head": "O(N_PT)", "full_ft": "O(N_PT)"},
    "Trainable Logit": {"head": "O(N_PT * T^2 * (H + V))", "full_ft": "O(N_PT * T^2 * (H + V))"},
    "Representation": {"head": "O(N_PT * H)", "full_ft": "O(N_PT * H)"},
    "Ground Truth": {"head": "O(N_PT * T * H * V)", "full_ft": "O(Fw(N))"},
}


@dataclass
class FlopEstimate:
    """Closed-form FLOP estimate for retrieving forgotten examples from D_PT."""

    n_upstream: int
    T: int
    H: int
    V: int
    candidate_vocab: int

    def threshold(self) -> float:
        return float(self.n_upstream)

    def representation(self) -> float:
        return float(self.n_upstream * self.H)

    def trainable_logit(self) -> float:
        return float(self.n_upstream * self.T * self.T * (self.H + self.V))

    def ground_truth(self, forward_flops_per_example: float) -> float:
        """``Fw(N_PT)``: one full forward pass of the updated LM over D_PT."""
        return float(self.n_upstream * forward_flops_per_example)

    def to_dict(self, forward_flops_per_example: float) -> Dict[str, float]:
        return {
            "Threshold": self.threshold(),
            "Representation": self.representation(),
            "Trainable Logit": self.trainable_logit(),
            "Ground Truth": self.ground_truth(forward_flops_per_example),
        }


def estimate_flops(
    n_upstream: int = 3600,
    T: int = 32,
    H: int = 1024,
    V: int = 32128,
    forward_flops_per_example: float = 2 * 780e6 * 64,
) -> Dict[str, float]:
    """FLOP counts in the style of Appendix C.

    ``forward_flops_per_example`` defaults to a rough estimate of one forward pass
    of FLAN-T5_Large on a 64-token sequence (2 * parameters * tokens).
    """
    return FlopEstimate(n_upstream, T, H, V, V).to_dict(forward_flops_per_example)
