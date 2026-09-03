"""
AntMaze evaluation task reward functions.

Implements the specific evaluation tasks described in the paper:
- Goal-reaching: 5 fixed goal locations
- Directional: 4 directional movement tasks
- Random simplex: 5 seeded procedural noise tasks
- Path tasks: center, loop, edges
"""

import numpy as np
import gym


class AntMazeGoalReaching:
    """
    Goal-reaching tasks for AntMaze.
    5 fixed goal locations with -1 reward until goal reached.
    """

    GOALS = {
        'goal-bottom': (28, 0),
        'goal-left': (0, 15),
        'goal-top': (35, 24),
        'goal-center': (12, 24),
        'goal-right': (33, 16)
    }

    def __init__(self, goal_name='goal-bottom', threshold=2.0):
        """
        Args:
            goal_name: One of the 5 goal names
            threshold: Distance threshold for considering goal reached
        """
        self.goal_name = goal_name
        self.goal_pos = np.array(self.GOALS[goal_name])
        self.threshold = threshold

    def __call__(self, state):
        """Compute reward for a state."""
        # Extract XY position from state (first 2 dimensions for AntMaze)
        xy_pos = state[..., :2]
        distance = np.linalg.norm(xy_pos - self.goal_pos, axis=-1)
        reward = np.where(distance < self.threshold, 0.0, -1.0)
        return reward

    @staticmethod
    def get_all_tasks():
        """Return list of all goal-reaching tasks."""
        return [AntMazeGoalReaching(name) for name in AntMazeGoalReaching.GOALS.keys()]


class AntMazeDirectional:
    """
    Directional movement tasks for AntMaze.
    Rewards agent for moving in a specific direction based on velocity.
    """

    DIRECTIONS = {
        'vel_left': (-1, 0),
        'vel_up': (0, 1),
        'vel_down': (0, -1),
        'vel_right': (1, 0)
    }

    def __init__(self, direction_name='vel_left'):
        """
        Args:
            direction_name: One of the 4 direction names
        """
        self.direction_name = direction_name
        self.target_velocity = np.array(self.DIRECTIONS[direction_name])

    def __call__(self, state, next_state=None):
        """
        Compute reward based on velocity alignment.

        Args:
            state: Current state
            next_state: Next state (to compute velocity)

        Returns:
            reward: Dot product between actual and target velocity
        """
        if next_state is None:
            # If no next state, return 0
            return 0.0

        # Compute velocity as change in XY position
        velocity = next_state[..., :2] - state[..., :2]

        # Reward is dot product with target velocity
        reward = np.sum(velocity * self.target_velocity, axis=-1)
        return reward

    @staticmethod
    def get_all_tasks():
        """Return list of all directional tasks."""
        return [AntMazeDirectional(name) for name in AntMazeDirectional.DIRECTIONS.keys()]


class AntMazeRandomSimplex:
    """
    Random simplex tasks using opensimplex noise for AntMaze.
    Creates a height map and velocity preferences based on procedural noise.
    """

    def __init__(self, seed=1):
        """
        Args:
            seed: Random seed (1-5 for the 5 evaluation tasks)
        """
        self.seed = seed
        try:
            from opensimplex import OpenSimplex
            self.noise = OpenSimplex(seed=seed)
        except ImportError:
            # Fallback to simple noise if opensimplex not available
            print("Warning: opensimplex not installed, using numpy random as fallback")
            self.rng = np.random.RandomState(seed)
            self.noise = None

    def __call__(self, state, next_state=None):
        """
        Compute reward based on height map and velocity preference.

        Args:
            state: Current state
            next_state: Next state (to compute velocity)

        Returns:
            reward: Baseline -1 + height bonus + velocity bonus
        """
        # Extract XY position
        xy_pos = state[..., :2]

        # Base reward
        reward = -1.0

        # Height bonus
        if self.noise is not None:
            height = self.noise.noise2(xy_pos[..., 0] / 10.0, xy_pos[..., 1] / 10.0)
        else:
            # Fallback: use deterministic noise based on position
            height = np.sin(xy_pos[..., 0] / 5.0 + self.seed) * np.cos(xy_pos[..., 1] / 5.0 + self.seed)

        reward += height * 0.5  # Scale height bonus

        # Velocity bonus
        if next_state is not None:
            velocity = next_state[..., :2] - xy_pos

            # Preferred velocity from noise field
            if self.noise is not None:
                pref_vx = self.noise.noise2(xy_pos[..., 0] / 10.0 + 100, xy_pos[..., 1] / 10.0)
                pref_vy = self.noise.noise2(xy_pos[..., 0] / 10.0, xy_pos[..., 1] / 10.0 + 100)
            else:
                pref_vx = np.cos(xy_pos[..., 0] / 5.0 + self.seed)
                pref_vy = np.sin(xy_pos[..., 1] / 5.0 + self.seed)

            pref_vel = np.stack([pref_vx, pref_vy], axis=-1)

            # Dot product bonus
            vel_bonus = np.sum(velocity * pref_vel, axis=-1)
            reward += vel_bonus * 0.5  # Scale velocity bonus

        return reward

    @staticmethod
    def get_all_tasks():
        """Return list of all random simplex tasks (5 seeds)."""
        return [AntMazeRandomSimplex(seed=i) for i in range(1, 6)]


class AntMazePath:
    """
    Hand-crafted path following tasks for AntMaze.
    Three variants: center corridor, loop around grid, edges of grid.
    """

    def __init__(self, path_type='center'):
        """
        Args:
            path_type: One of 'center', 'loop', 'edges'
        """
        self.path_type = path_type

    def _center_corridor_reward(self, xy_pos):
        """Reward for following center corridor."""
        # Define center corridor as central region of the maze
        center_x = 17.5  # Middle of 0-35
        corridor_width = 8.0
        distance_from_center = np.abs(xy_pos[..., 0] - center_x)
        reward = np.where(distance_from_center < corridor_width, 1.0, -1.0)
        return reward

    def _loop_reward(self, xy_pos, next_pos=None):
        """Reward for moving in a loop around the grid."""
        # Define a loop trajectory (counterclockwise)
        # Simplified: reward based on being near the edges and moving counterclockwise

        x, y = xy_pos[..., 0], xy_pos[..., 1]

        # Check if near edges
        near_edge = ((x < 5) | (x > 30) | (y < 5) | (y > 19))

        # Base reward for being near edge
        reward = np.where(near_edge, 1.0, -1.0)

        # Bonus for moving counterclockwise
        if next_pos is not None:
            dx = next_pos[..., 0] - x
            dy = next_pos[..., 1] - y

            # Determine quadrant and preferred direction
            # Top edge: move left
            # Left edge: move down
            # Bottom edge: move right
            # Right edge: move up
            top = (y > 18) & (dy < 0)
            left = (x < 5) & (dx < 0)
            bottom = (y < 6) & (dy > 0)
            right = (x > 30) & (dx > 0)

            correct_direction = top | left | bottom | right
            reward = np.where(correct_direction, reward + 0.5, reward)

        return reward

    def _edges_reward(self, xy_pos):
        """Reward for moving along the edges of the grid."""
        x, y = xy_pos[..., 0], xy_pos[..., 1]

        # Near any edge
        edge_threshold = 5.0
        near_edge = ((x < edge_threshold) | (x > 35 - edge_threshold) |
                     (y < edge_threshold) | (y > 24 - edge_threshold))

        reward = np.where(near_edge, 1.0, -1.0)
        return reward

    def __call__(self, state, next_state=None):
        """Compute reward based on path type."""
        xy_pos = state[..., :2]
        next_pos = next_state[..., :2] if next_state is not None else None

        if self.path_type == 'center':
            return self._center_corridor_reward(xy_pos)
        elif self.path_type == 'loop':
            return self._loop_reward(xy_pos, next_pos)
        elif self.path_type == 'edges':
            return self._edges_reward(xy_pos)
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

    @staticmethod
    def get_all_tasks():
        """Return list of all path tasks."""
        return [AntMazePath(path_type) for path_type in ['center', 'loop', 'edges']]
