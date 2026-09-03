"""
ExORL evaluation task reward functions.

Implements the specific evaluation tasks for Walker and Cheetah domains
as described in the paper addendum.
"""

import numpy as np


class CheetahVelocityTask:
    """
    Cheetah velocity tasks.
    Rewards agent for achieving target forward/backward velocity.
    """

    def __init__(self, task_name='run', backward=False):
        """
        Args:
            task_name: 'run' (velocity >= 10) or 'walk' (velocity >= 1)
            backward: If True, reward backward velocity instead
        """
        self.task_name = task_name
        self.backward = backward

        # Set velocity threshold
        if task_name == 'run':
            self.threshold = 10.0
        elif task_name == 'walk':
            self.threshold = 1.0
        else:
            raise ValueError(f"Unknown task: {task_name}")

        self.direction = -1.0 if backward else 1.0

    def __call__(self, state, velocity=None):
        """
        Compute reward based on horizontal velocity.

        Args:
            state: Current state
            velocity: Horizontal velocity (if not in state)

        Returns:
            reward: 1 if velocity >= threshold, linear decay below
        """
        # If velocity is provided directly, use it
        if velocity is not None:
            horiz_vel = velocity
        else:
            # Assume velocity is in the state (needs physics info)
            # This is a simplified version - in practice, velocity
            # would be computed from physics or appended to state
            horiz_vel = state[..., -1] if state.shape[-1] > 0 else 0.0

        # Apply direction (forward or backward)
        directed_vel = horiz_vel * self.direction

        # Compute reward
        if directed_vel >= self.threshold:
            reward = 1.0
        elif directed_vel > 0:
            # Linear decay from threshold to 0
            reward = directed_vel / self.threshold
        else:
            # Opposite direction
            reward = 0.0

        return reward

    @staticmethod
    def get_all_tasks():
        """Return list of all cheetah velocity tasks."""
        return [
            CheetahVelocityTask('run', backward=False),
            CheetahVelocityTask('run', backward=True),
            CheetahVelocityTask('walk', backward=False),
            CheetahVelocityTask('walk', backward=True),
        ]


class WalkerVelocityTask:
    """
    Walker velocity tasks.
    Rewards agent for achieving velocity at or above threshold.
    """

    THRESHOLDS = [0.1, 1.0, 4.0, 8.0]

    def __init__(self, threshold=1.0):
        """
        Args:
            threshold: Minimum velocity to achieve max reward
        """
        self.threshold = threshold

    def __call__(self, state, velocity=None):
        """
        Compute reward based on horizontal velocity.

        Args:
            state: Current state
            velocity: Horizontal velocity (if not in state)

        Returns:
            reward: 1 if velocity >= threshold, linear decay below
        """
        # If velocity is provided directly, use it
        if velocity is not None:
            horiz_vel = velocity
        else:
            # Assume velocity is in the state
            horiz_vel = state[..., -1] if state.shape[-1] > 0 else 0.0

        # Compute reward
        if horiz_vel >= self.threshold:
            reward = 1.0
        elif horiz_vel > 0:
            reward = horiz_vel / self.threshold
        else:
            reward = 0.0

        return reward

    @staticmethod
    def get_all_tasks():
        """Return list of all walker velocity tasks."""
        return [WalkerVelocityTask(t) for t in WalkerVelocityTask.THRESHOLDS]


class GoalReachingTask:
    """
    Goal-reaching task for ExORL domains.
    5 random states from dataset used as goals.
    """

    def __init__(self, goal_state, threshold=0.1):
        """
        Args:
            goal_state: Target goal state
            threshold: Distance threshold (Euclidean)
        """
        self.goal_state = goal_state
        self.threshold = threshold

    def __call__(self, state):
        """
        Compute reward based on distance to goal.

        Args:
            state: Current state

        Returns:
            reward: 0 if within threshold, -1 otherwise
        """
        # Compute Euclidean distance
        # Note: In paper, states are normalized by std before distance computation
        distance = np.linalg.norm(state - self.goal_state)

        reward = 0.0 if distance < self.threshold else -1.0
        return reward

    @staticmethod
    def create_from_dataset(dataset, num_goals=5, seed=0):
        """
        Create goal-reaching tasks from random states in dataset.

        Args:
            dataset: Dataset containing states
            num_goals: Number of goal states to sample
            seed: Random seed for reproducibility

        Returns:
            List of GoalReachingTask instances
        """
        rng = np.random.RandomState(seed)
        states = dataset['observations']

        # Sample random states as goals
        goal_indices = rng.choice(len(states), size=num_goals, replace=False)
        goal_states = states[goal_indices]

        tasks = [GoalReachingTask(goal) for goal in goal_states]
        return tasks
