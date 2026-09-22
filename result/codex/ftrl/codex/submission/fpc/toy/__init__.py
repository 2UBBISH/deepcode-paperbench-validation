"""Toy environments used to build intuition about forgetting of pre-trained
capabilities (Appendix A).

* :mod:`fpc.toy.two_state_mdp` -- the two-state MDP from Appendix A.1 showing
  that the state coverage gap and the imperfect cloning gap already appear in a
  two-state MDP.
* :mod:`fpc.toy.apple_retrieval` -- the AppleRetrieval grid-world from
  Appendix A.2, a REINFORCE experiment with a linear policy showing that FPC
  grows with the length of the CLOSE phase.
"""

from .two_state_mdp import TwoStateMDP, state_coverage_gap_f, imperfect_cloning_gap_f
from .apple_retrieval import AppleRetrieval, LinearPolicy, reinforce

__all__ = [
    "TwoStateMDP",
    "state_coverage_gap_f",
    "imperfect_cloning_gap_f",
    "AppleRetrieval",
    "LinearPolicy",
    "reinforce",
]
