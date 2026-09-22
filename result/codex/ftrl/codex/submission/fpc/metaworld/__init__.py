"""RoboticSequence experiments based on Meta-World (Section 3-5, Appendix B.3, F).

Setting
-------
``RoboticSequence`` chains several Meta-World tasks into a single episode.  The
robot is successful only if, within one episode, it completes the sub-tasks in
order.  The main-text sequence is::

    hammer -> push -> peg-unplug-side -> push-wall

The pre-trained policy ``pi_*`` solves the last two stages
(``peg-unplug-side`` and ``push-wall``, the FAR states) but not the first two
(``hammer`` and ``push``, the CLOSE states), producing a state coverage gap.

Algorithms
----------
* :mod:`fpc.metaworld.sac` -- Soft Actor-Critic (Haarnoja et al., 2018a) with a
  separate head per stage and automatic entropy tuning.
* :mod:`fpc.metaworld.retention` -- EWC / BC / EM glue for SAC.
* :mod:`fpc.metaworld.robotic_sequence` -- the environment wrapper.
* :mod:`fpc.metaworld.eval` -- per-stage success rates, forward transfer.
* :mod:`fpc.metaworld.cka` -- the representation-similarity analysis.
"""
