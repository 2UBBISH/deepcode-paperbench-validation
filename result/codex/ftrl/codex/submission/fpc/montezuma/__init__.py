"""Montezuma's Revenge experiments (Section 3-5, Appendix B.2, E).

Setting
-------
* Pre-training: PPO with Random Network Distillation (RND) is trained until it
  reaches an episode cumulative reward of ~7000 (addendum).  500 trajectories
  are then collected and used by the behavioral-cloning loss.
* Pre-training is restricted to the rooms *from a certain room onward* -- in the
  main text, Room 7 onward.  Rooms 1-6 are therefore CLOSE states, Room 7 and
  beyond are FAR states.
* Fine-tuning: the agent starts from Room 1 and has to solve the whole game.
* Knowledge retention: BC (and EWC) mitigate the state coverage gap.
"""
