"""Downstream evaluation tasks for the three benchmark domains.

The paper evaluates FRE zero-shot on:

  * AntMaze  (Section 5, Figure 4, Table 1) -- ``fre.tasks.antmaze``
  * ExORL    (walker + cheetah)             -- ``fre.tasks.exorl``
  * Kitchen                                 -- ``fre.tasks.kitchen``

Every task is expressed as a reward function over environment states so that
the same 32 ``(state, reward)`` samples used to condition the FRE agent come
straight from the task definition.
"""

from fre.tasks.base import EvalTask, TaskSuite, evaluate_policy_on_task

__all__ = ["EvalTask", "TaskSuite", "evaluate_policy_on_task"]
