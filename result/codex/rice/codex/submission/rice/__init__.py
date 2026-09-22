"""RICE: Refining reinforcement learning agents with explanation.

Reproduction of

    "RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning
     with Explanation" (Cheng, Wu, Yu, Yang, Wang, Xing; ICML 2024).

The package is organised following the two algorithmic contributions of the
paper:

* :mod:`rice.explanation` -- Algorithm 1: training a *state mask network* that
  yields step level importance scores (the RICE variant of StateMask).
* :mod:`rice.refining`   -- Algorithm 2: refining a pre-trained agent from a
  *mixed initial state distribution* made of the default initial states and the
  critical states identified by the mask network, together with an RND
  exploration bonus.

See ``README.md`` for the mapping between paper sections/figures and the code.
"""

__version__ = "0.1.0"
