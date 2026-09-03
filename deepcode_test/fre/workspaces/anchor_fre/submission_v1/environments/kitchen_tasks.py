"""
Kitchen evaluation task reward functions.

The D4RL Kitchen environment has 7 standard subtasks,
each with its own sparse reward function already defined.
"""

import numpy as np


class KitchenTask:
    """
    Wrapper for Kitchen subtasks.

    The Kitchen environment in D4RL has 7 subtasks:
    - bottom burner
    - top burner
    - light switch
    - slide cabinet
    - hinge cabinet
    - microwave
    - kettle

    Each task has a sparse reward (0 or 1) for completion.
    """

    TASK_ELEMENTS = [
        'bottom burner',
        'top burner',
        'light switch',
        'slide cabinet',
        'hinge cabinet',
        'microwave',
        'kettle'
    ]

    def __init__(self, task_name):
        """
        Args:
            task_name: One of the 7 Kitchen subtask names
        """
        if task_name not in self.TASK_ELEMENTS:
            raise ValueError(f"Unknown task: {task_name}. "
                           f"Must be one of {self.TASK_ELEMENTS}")

        self.task_name = task_name
        self.task_idx = self.TASK_ELEMENTS.index(task_name)

    def __call__(self, state, obs_dict=None):
        """
        Compute reward for Kitchen task.

        Args:
            state: State observation (may include task completion flags)
            obs_dict: Optional dictionary with task-specific information

        Returns:
            reward: 1.0 if task completed, 0.0 otherwise
        """
        # In D4RL Kitchen, the environment tracks completion of each subtask
        # The reward is typically computed by the environment itself

        # This is a simplified placeholder that would need to be
        # integrated with the actual Kitchen environment's reward computation
        if obs_dict is not None and 'task_completion' in obs_dict:
            # If we have task completion info
            return float(obs_dict['task_completion'][self.task_idx])
        else:
            # Fallback: return 0 (would need environment integration)
            return 0.0

    @staticmethod
    def get_all_tasks():
        """Return list of all Kitchen subtasks."""
        return [KitchenTask(name) for name in KitchenTask.TASK_ELEMENTS]


class KitchenMultiTask:
    """
    Combined Kitchen task that rewards completing any of multiple subtasks.
    """

    def __init__(self, task_names=None):
        """
        Args:
            task_names: List of subtask names to include (default: all 7)
        """
        if task_names is None:
            task_names = KitchenTask.TASK_ELEMENTS

        self.tasks = [KitchenTask(name) for name in task_names]
        self.num_tasks = len(self.tasks)

    def __call__(self, state, obs_dict=None):
        """
        Compute reward as sum of completed subtasks.

        Args:
            state: State observation
            obs_dict: Optional dictionary with task-specific information

        Returns:
            reward: Number of completed tasks
        """
        total_reward = sum(task(state, obs_dict) for task in self.tasks)
        return total_reward / self.num_tasks  # Normalize to [0, 1]
