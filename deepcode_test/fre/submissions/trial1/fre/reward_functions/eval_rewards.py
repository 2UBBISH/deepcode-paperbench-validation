"""
Evaluation reward functions for all downstream benchmark tasks.

Implements the reward functions used for zero-shot evaluation on:
- AntMaze: goal-reaching, directional, random-simplex (Perlin noise), path tasks
- ExORL (Walker, Cheetah): goal-reaching, velocity
- Kitchen: 7 subtask completion indicators

All reward functions inherit from RewardFunction base class and provide
a consistent interface for evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from fre.reward_functions.base import RewardFunction


# ============================================================
# AntMaze Reward Functions
# ============================================================

class AntMazeGoalReward(RewardFunction):
    """
    Goal-reaching reward for AntMaze: 0 if within epsilon of goal, -1 otherwise.
    The goal is a specific (x, y) location in the maze.
    """
    
    def __init__(
        self,
        state_dim: int = 29,
        goal_x: float = 0.0,
        goal_y: float = 0.0,
        epsilon: float = 0.5,
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.epsilon = epsilon
        # AntMaze state: first two dimensions are (x, y) position
        self.register_buffer('goal', torch.tensor([goal_x, goal_y], 
                                                   dtype=torch.float32))
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor, 0 if within epsilon, -1 otherwise
        """
        states = states.to(self.device)
        # Use first 2 dimensions (x, y position)
        positions = states[:, :2]
        distances = torch.norm(positions - self.goal.unsqueeze(0), dim=-1)
        rewards = torch.where(distances < self.epsilon, 
                              torch.zeros_like(distances), 
                              -torch.ones_like(distances))
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'antmaze_goal',
            'goal': self.goal.cpu().tolist(),
            'epsilon': self.epsilon
        }


class AntMazeDirectionalReward(RewardFunction):
    """
    Directional reward for AntMaze: dot product of desired direction with velocity.
    Rewards the agent for moving in a specific direction.
    """
    
    def __init__(
        self,
        state_dim: int = 29,
        direction_x: float = 1.0,
        direction_y: float = 0.0,
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.register_buffer('direction', 
                             torch.tensor([direction_x, direction_y], 
                                          dtype=torch.float32))
        # Normalize direction
        self.direction = self.direction / (self.direction.norm() + 1e-8)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor, dot product of velocity with direction
        """
        states = states.to(self.device)
        # AntMaze state: indices 2-4 are velocities (vx, vy, vz) or similar
        # Typically indices 2 and 3 are x and y velocities
        velocities = states[:, 2:4]  # (vx, vy)
        rewards = torch.sum(velocities * self.direction.unsqueeze(0), dim=-1)
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'antmaze_directional',
            'direction': self.direction.cpu().tolist()
        }


class AntMazeRandomSimplexReward(RewardFunction):
    """
    Random simplex (Perlin-noise-like) reward function over the AntMaze xy-plane.
    Uses a random Fourier feature approximation to generate smooth random functions
    over 2D space, mimicking the procedural noise generator described in the paper.
    """
    
    def __init__(
        self,
        state_dim: int = 29,
        num_features: int = 100,
        scale: float = 1.0,
        seed: int = 0,
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.num_features = num_features
        self.scale = scale
        
        # Generate random Fourier features
        rng = np.random.RandomState(seed)
        # Random frequencies
        self.register_buffer('frequencies', 
                             torch.tensor(rng.randn(num_features, 2) * scale, 
                                          dtype=torch.float32))
        # Random phases
        self.register_buffer('phases', 
                             torch.tensor(rng.uniform(0, 2 * np.pi, num_features), 
                                          dtype=torch.float32))
        # Random coefficients
        self.register_buffer('coefficients', 
                             torch.tensor(rng.randn(num_features) / np.sqrt(num_features), 
                                          dtype=torch.float32))
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor
        """
        states = states.to(self.device)
        positions = states[:, :2]  # (x, y)
        
        # Compute Fourier features: cos(2*pi * (freq · pos) + phase)
        projections = torch.matmul(positions, self.frequencies.T)  # (B, num_features)
        features = torch.cos(2 * np.pi * projections + self.phases.unsqueeze(0))
        
        # Weighted sum
        rewards = torch.matmul(features, self.coefficients)
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'antmaze_random_simplex',
            'num_features': self.num_features,
            'scale': self.scale
        }


class AntMazePathReward(RewardFunction):
    """
    Path-based reward for AntMaze: rewards the agent for being near a specified
    path (sequence of waypoints). Used for edges, loop, and center tasks.
    
    Reward = -min distance to any point on the path.
    """
    
    def __init__(
        self,
        state_dim: int = 29,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        path_type: str = "edges",  # "edges", "loop", "center"
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.path_type = path_type
        
        # Define waypoints based on path type
        if waypoints is None:
            waypoints = self._get_default_waypoints(path_type)
        
        self.register_buffer('waypoints', 
                             torch.tensor(waypoints, dtype=torch.float32))
        self.num_waypoints = len(waypoints)
        
    def _get_default_waypoints(self, path_type: str) -> List[Tuple[float, float]]:
        """Get default waypoints for each path type."""
        if path_type == "edges":
            # Path along the edges of the maze
            return [
                (0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0),
                (6.0, 2.0), (6.0, 4.0), (6.0, 6.0),
                (4.0, 6.0), (2.0, 6.0), (0.0, 6.0),
                (0.0, 4.0), (0.0, 2.0), (0.0, 0.0)
            ]
        elif path_type == "loop":
            # A loop in the center of the maze
            return [
                (2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0), (2.0, 2.0)
            ]
        elif path_type == "center":
            # Path through the center
            return [
                (0.0, 3.0), (2.0, 3.0), (4.0, 3.0), (6.0, 3.0)
            ]
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor, negative minimum distance to path
        """
        states = states.to(self.device)
        positions = states[:, :2]  # (B, 2)
        
        # Compute distances to all waypoints
        # positions: (B, 2), waypoints: (W, 2)
        # distances: (B, W)
        distances = torch.cdist(positions, self.waypoints.unsqueeze(0).expand(
            positions.shape[0], -1, -1).squeeze(0) if positions.shape[0] == 1 
            else torch.cdist(positions, self.waypoints))
        
        # Also compute distances to line segments between consecutive waypoints
        if self.num_waypoints > 1:
            seg_distances = []
            for i in range(self.num_waypoints - 1):
                p1 = self.waypoints[i]
                p2 = self.waypoints[i + 1]
                seg_dist = self._point_to_segment_distance(positions, p1, p2)
                seg_distances.append(seg_dist)
            seg_distances = torch.stack(seg_distances, dim=-1)  # (B, W-1)
            all_distances = torch.cat([distances, seg_distances], dim=-1)
        else:
            all_distances = distances
        
        min_distances = torch.min(all_distances, dim=-1)[0]
        rewards = -min_distances
        return rewards
    
    def _point_to_segment_distance(
        self, 
        points: torch.Tensor,  # (B, 2)
        p1: torch.Tensor,      # (2,)
        p2: torch.Tensor       # (2,)
    ) -> torch.Tensor:
        """Compute minimum distance from points to line segment p1-p2."""
        segment = p2 - p1
        segment_len_sq = torch.dot(segment, segment)
        
        if segment_len_sq < 1e-8:
            return torch.norm(points - p1.unsqueeze(0), dim=-1)
        
        # Project points onto line
        t = torch.sum((points - p1.unsqueeze(0)) * segment.unsqueeze(0), dim=-1) / segment_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        
        # Closest point on segment
        projection = p1.unsqueeze(0) + t.unsqueeze(-1) * segment.unsqueeze(0)
        
        return torch.norm(points - projection, dim=-1)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': f'antmaze_path_{self.path_type}',
            'path_type': self.path_type,
            'num_waypoints': self.num_waypoints
        }


# ============================================================
# ExORL Reward Functions
# ============================================================

class ExORLGoalReward(RewardFunction):
    """
    Goal-reaching reward for ExORL (Walker/Cheetah): 
    0 if within epsilon of goal state, -1 otherwise.
    Uses a subset of state dimensions for distance computation.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_state: Optional[torch.Tensor] = None,
        epsilon: float = 0.1,
        use_all_dims: bool = False,
        relevant_dims: Optional[List[int]] = None,
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.epsilon = epsilon
        self.use_all_dims = use_all_dims
        
        if relevant_dims is not None:
            self.relevant_dims = relevant_dims
        elif use_all_dims:
            self.relevant_dims = list(range(state_dim))
        else:
            # Default: use position-related dimensions (first few dims)
            # For Walker/Cheetah, typically first 2-3 dims are positions
            self.relevant_dims = list(range(min(8, state_dim)))
        
        if goal_state is not None:
            self.register_buffer('goal_state', goal_state.to(torch.float32))
        else:
            self.register_buffer('goal_state', torch.zeros(state_dim))
            
    def set_goal(self, goal_state: torch.Tensor):
        """Set the goal state."""
        self.goal_state = goal_state.to(self.device).to(torch.float32)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor
        """
        states = states.to(self.device)
        goal = self.goal_state.to(self.device)
        
        # Use only relevant dimensions for distance
        states_sub = states[:, self.relevant_dims]
        goal_sub = goal[self.relevant_dims]
        
        distances = torch.norm(states_sub - goal_sub.unsqueeze(0), dim=-1)
        rewards = torch.where(distances < self.epsilon,
                              torch.zeros_like(distances),
                              -torch.ones_like(distances))
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'exorl_goal',
            'epsilon': self.epsilon,
            'relevant_dims': self.relevant_dims
        }


class ExORLVelocityReward(RewardFunction):
    """
    Velocity reward for ExORL (Walker/Cheetah):
    Rewards the agent for achieving a target velocity.
    Reward = -|current_velocity - target_velocity|
    or simply the velocity in a given direction.
    """
    
    def __init__(
        self,
        state_dim: int,
        target_velocity: float = 1.0,
        velocity_idx: int = 0,  # Index of velocity in state
        mode: str = "forward",  # "forward", "backward", or "target"
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        self.target_velocity = target_velocity
        self.velocity_idx = velocity_idx
        self.mode = mode
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor
        """
        states = states.to(self.device)
        velocity = states[:, self.velocity_idx]
        
        if self.mode == "forward":
            rewards = velocity
        elif self.mode == "backward":
            rewards = -velocity
        elif self.mode == "target":
            rewards = -torch.abs(velocity - self.target_velocity)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'exorl_velocity',
            'target_velocity': self.target_velocity,
            'velocity_idx': self.velocity_idx,
            'mode': self.mode
        }


# ============================================================
# Kitchen Reward Functions
# ============================================================

class KitchenSubtaskReward(RewardFunction):
    """
    Binary subtask completion reward for Kitchen environment.
    The Kitchen environment has 7 subtasks, each indicated by specific
    state dimensions exceeding certain thresholds.
    
    Subtasks (based on D4RL kitchen environment):
    0: Microwave open
    1: Light switch on
    2: Slide cabinet open
    3: Hinge cabinet open
    4: Bottom burner on
    5: Top burner on
    6: Kettle on
    """
    
    # Known subtask thresholds for Kitchen environment
    SUBTASK_NAMES = [
        "microwave",
        "light_switch",
        "slide_cabinet",
        "hinge_cabinet",
        "bottom_burner",
        "top_burner",
        "kettle"
    ]
    
    def __init__(
        self,
        state_dim: int = 60,
        subtask_idx: int = 0,
        threshold: float = 0.5,
        state_idx: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: Dimension of state space
            subtask_idx: Which subtask (0-6) to reward
            threshold: Threshold for binary completion
            state_idx: Specific state dimension index for this subtask.
                      If None, uses subtask_idx as the index.
        """
        super().__init__(state_dim, device)
        self.subtask_idx = subtask_idx
        self.threshold = threshold
        self.state_idx = state_idx if state_idx is not None else subtask_idx
        self.subtask_name = self.SUBTASK_NAMES[subtask_idx] if subtask_idx < 7 else f"subtask_{subtask_idx}"
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor, 1.0 if subtask completed, 0.0 otherwise
        """
        states = states.to(self.device)
        # The subtask completion is indicated by a specific state dimension
        subtask_value = states[:, self.state_idx]
        rewards = (subtask_value > self.threshold).float()
        return rewards
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'kitchen_subtask',
            'subtask_idx': self.subtask_idx,
            'subtask_name': self.subtask_name,
            'threshold': self.threshold,
            'state_idx': self.state_idx
        }


class KitchenAllSubtasksReward(RewardFunction):
    """
    Reward function that sums all 7 Kitchen subtask completions.
    This is the standard evaluation metric for Kitchen: number of completed subtasks.
    """
    
    def __init__(
        self,
        state_dim: int = 60,
        thresholds: Optional[List[float]] = None,
        state_indices: Optional[List[int]] = None,
        device: Optional[str] = None
    ):
        super().__init__(state_dim, device)
        
        if thresholds is None:
            thresholds = [0.5] * 7
        if state_indices is None:
            state_indices = list(range(7))
            
        self.num_subtasks = len(thresholds)
        self.register_buffer('thresholds', torch.tensor(thresholds, dtype=torch.float32))
        self.state_indices = state_indices
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch_size, state_dim) tensor
        Returns:
            rewards: (batch_size,) tensor, sum of completed subtasks (0-7)
        """
        states = states.to(self.device)
        total_reward = torch.zeros(states.shape[0], device=self.device)
        
        for i, idx in enumerate(self.state_indices):
            subtask_value = states[:, idx]
            completed = (subtask_value > self.thresholds[i]).float()
            total_reward += completed
            
        return total_reward
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'type': 'kitchen_all_subtasks',
            'num_subtasks': self.num_subtasks,
            'state_indices': self.state_indices
        }


# ============================================================
# Evaluation Reward Function Factory
# ============================================================

def create_eval_reward_function(
    task_name: str,
    state_dim: int,
    device: Optional[str] = None,
    **kwargs
) -> RewardFunction:
    """
    Factory function to create evaluation reward functions by task name.
    
    Supported tasks:
        AntMaze:
            - 'antmaze_goal_<x>_<y>' (e.g., 'antmaze_goal_2.0_4.0')
            - 'antmaze_directional_<dx>_<dy>' (e.g., 'antmaze_directional_1.0_0.0')
            - 'antmaze_random_simplex_<seed>'
            - 'antmaze_path_edges', 'antmaze_path_loop', 'antmaze_path_center'
        
        ExORL:
            - 'exorl_goal_<domain>' (e.g., 'exorl_goal_walker')
            - 'exorl_velocity_<domain>_<direction>' (e.g., 'exorl_velocity_walker_forward')
        
        Kitchen:
            - 'kitchen_subtask_<idx>' (e.g., 'kitchen_subtask_0')
            - 'kitchen_all'
    
    Args:
        task_name: Name of the evaluation task
        state_dim: State dimension
        device: Device to place tensors on
        **kwargs: Additional arguments for specific reward functions
    
    Returns:
        RewardFunction instance
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse task name
    parts = task_name.split('_')
    
    if parts[0] == 'antmaze':
        if parts[1] == 'goal':
            # antmaze_goal_<x>_<y>
            goal_x = float(parts[2]) if len(parts) > 2 else kwargs.get('goal_x', 0.0)
            goal_y = float(parts[3]) if len(parts) > 3 else kwargs.get('goal_y', 0.0)
            epsilon = kwargs.get('epsilon', 0.5)
            return AntMazeGoalReward(
                state_dim=state_dim,
                goal_x=goal_x,
                goal_y=goal_y,
                epsilon=epsilon,
                device=device
            )
        elif parts[1] == 'directional':
            # antmaze_directional_<dx>_<dy>
            dx = float(parts[2]) if len(parts) > 2 else kwargs.get('dx', 1.0)
            dy = float(parts[3]) if len(parts) > 3 else kwargs.get('dy', 0.0)
            return AntMazeDirectionalReward(
                state_dim=state_dim,
                direction_x=dx,
                direction_y=dy,
                device=device
            )
        elif parts[1] == 'random' and parts[2] == 'simplex':
            seed = int(parts[3]) if len(parts) > 3 else kwargs.get('seed', 0)
            return AntMazeRandomSimplexReward(
                state_dim=state_dim,
                seed=seed,
                device=device,
                **{k: v for k, v in kwargs.items() if k != 'seed'}
            )
        elif parts[1] == 'path':
            path_type = parts[2] if len(parts) > 2 else kwargs.get('path_type', 'edges')
            return AntMazePathReward(
                state_dim=state_dim,
                path_type=path_type,
                device=device
            )
    
    elif parts[0] == 'exorl':
        if parts[1] == 'goal':
            domain = parts[2] if len(parts) > 2 else kwargs.get('domain', 'walker')
            epsilon = kwargs.get('epsilon', 0.1)
            return ExORLGoalReward(
                state_dim=state_dim,
                epsilon=epsilon,
                device=device
            )
        elif parts[1] == 'velocity':
            domain = parts[2] if len(parts) > 2 else kwargs.get('domain', 'walker')
            direction = parts[3] if len(parts) > 3 else kwargs.get('direction', 'forward')
            velocity_idx = kwargs.get('velocity_idx', 0)
            if direction == 'forward':
                mode = 'forward'
            elif direction == 'backward':
                mode = 'backward'
            else:
                mode = 'target'
            return ExORLVelocityReward(
                state_dim=state_dim,
                target_velocity=kwargs.get('target_velocity', 1.0),
                velocity_idx=velocity_idx,
                mode=mode,
                device=device
            )
    
    elif parts[0] == 'kitchen':
        if parts[1] == 'subtask':
            subtask_idx = int(parts[2]) if len(parts) > 2 else kwargs.get('subtask_idx', 0)
            return KitchenSubtaskReward(
                state_dim=state_dim,
                subtask_idx=subtask_idx,
                threshold=kwargs.get('threshold', 0.5),
                state_idx=kwargs.get('state_idx', subtask_idx),
                device=device
            )
        elif parts[1] == 'all':
            return KitchenAllSubtasksReward(
                state_dim=state_dim,
                device=device,
                **{k: v for k, v in kwargs.items() if k in ['thresholds', 'state_indices']}
            )
    
    raise ValueError(f"Unknown task name: {task_name}")


# ============================================================
# Pre-defined evaluation task lists
# ============================================================

ANTMAZE_EVAL_TASKS = [
    # Goal-reaching tasks (various goal locations)
    'antmaze_goal_2.0_4.0',
    'antmaze_goal_4.0_2.0',
    'antmaze_goal_6.0_4.0',
    'antmaze_goal_4.0_6.0',
    'antmaze_goal_0.0_0.0',
    'antmaze_goal_6.0_6.0',
    # Directional tasks
    'antmaze_directional_1.0_0.0',   # East
    'antmaze_directional_-1.0_0.0',  # West
    'antmaze_directional_0.0_1.0',   # North
    'antmaze_directional_0.0_-1.0',  # South
    # Random simplex tasks
    'antmaze_random_simplex_0',
    'antmaze_random_simplex_1',
    'antmaze_random_simplex_2',
    # Path tasks
    'antmaze_path_edges',
    'antmaze_path_loop',
    'antmaze_path_center',
]

EXORL_WALKER_EVAL_TASKS = [
    'exorl_goal_walker',
    'exorl_velocity_walker_forward',
    'exorl_velocity_walker_backward',
]

EXORL_CHEETAH_EVAL_TASKS = [
    'exorl_goal_cheetah',
    'exorl_velocity_cheetah_forward',
    'exorl_velocity_cheetah_backward',
]

KITCHEN_EVAL_TASKS = [
    'kitchen_subtask_0',  # Microwave
    'kitchen_subtask_1',  # Light switch
    'kitchen_subtask_2',  # Slide cabinet
    'kitchen_subtask_3',  # Hinge cabinet
    'kitchen_subtask_4',  # Bottom burner
    'kitchen_subtask_5',  # Top burner
    'kitchen_subtask_6',  # Kettle
    'kitchen_all',        # All subtasks combined
]


def get_eval_tasks(domain: str) -> List[str]:
    """
    Get the list of evaluation task names for a given domain.
    
    Args:
        domain: One of 'antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen'
    
    Returns:
        List of task name strings
    """
    domain_lower = domain.lower()
    if domain_lower == 'antmaze':
        return ANTMAZE_EVAL_TASKS
    elif domain_lower in ('exorl_walker', 'walker'):
        return EXORL_WALKER_EVAL_TASKS
    elif domain_lower in ('exorl_cheetah', 'cheetah'):
        return EXORL_CHEETAH_EVAL_TASKS
    elif domain_lower == 'kitchen':
        return KITCHEN_EVAL_TASKS
    else:
        raise ValueError(f"Unknown domain: {domain}. "
                         f"Expected one of: antmaze, exorl_walker, exorl_cheetah, kitchen")


# ============================================================
# Testing
# ============================================================

def test_eval_rewards():
    """Quick test of evaluation reward functions."""
    print("Testing evaluation reward functions...")
    
    device = 'cpu'
    
    # Test AntMaze goal
    reward_fn = AntMazeGoalReward(state_dim=29, goal_x=2.0, goal_y=4.0, device=device)
    states = torch.randn(10, 29)
    states[0, :2] = torch.tensor([2.0, 4.0])  # At goal
    states[1, :2] = torch.tensor([2.1, 4.1])  # Near goal
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    assert rewards[0] == 0.0, f"Expected 0 at goal, got {rewards[0]}"
    print(f"  AntMaze goal: OK (rewards: {rewards[:3]})")
    
    # Test AntMaze directional
    reward_fn = AntMazeDirectionalReward(state_dim=29, direction_x=1.0, direction_y=0.0, device=device)
    states = torch.randn(10, 29)
    states[0, 2:4] = torch.tensor([1.0, 0.0])  # Moving east
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    print(f"  AntMaze directional: OK (rewards: {rewards[:3]})")
    
    # Test AntMaze random simplex
    reward_fn = AntMazeRandomSimplexReward(state_dim=29, seed=42, device=device)
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    print(f"  AntMaze random simplex: OK (rewards: {rewards[:3]})")
    
    # Test AntMaze path
    for path_type in ['edges', 'loop', 'center']:
        reward_fn = AntMazePathReward(state_dim=29, path_type=path_type, device=device)
        rewards = reward_fn(states)
        assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
        print(f"  AntMaze path {path_type}: OK (rewards: {rewards[:3]})")
    
    # Test ExORL goal
    reward_fn = ExORLGoalReward(state_dim=24, device=device)
    goal = torch.randn(24)
    reward_fn.set_goal(goal)
    states = torch.randn(10, 24)
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    print(f"  ExORL goal: OK (rewards: {rewards[:3]})")
    
    # Test ExORL velocity
    reward_fn = ExORLVelocityReward(state_dim=24, mode='forward', device=device)
    states = torch.randn(10, 24)
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    print(f"  ExORL velocity forward: OK (rewards: {rewards[:3]})")
    
    # Test Kitchen subtask
    reward_fn = KitchenSubtaskReward(state_dim=60, subtask_idx=0, device=device)
    states = torch.rand(10, 60)
    states[0, 0] = 0.8  # Completed
    states[1, 0] = 0.2  # Not completed
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"
    assert rewards[1] == 0.0, f"Expected 0.0, got {rewards[1]}"
    print(f"  Kitchen subtask: OK (rewards: {rewards[:3]})")
    
    # Test Kitchen all subtasks
    reward_fn = KitchenAllSubtasksReward(state_dim=60, device=device)
    states = torch.rand(10, 60)
    states[0, :7] = 0.8  # All completed
    rewards = reward_fn(states)
    assert rewards.shape == (10,), f"Expected shape (10,), got {rewards.shape}"
    assert rewards[0] == 7.0, f"Expected 7.0, got {rewards[0]}"
    print(f"  Kitchen all subtasks: OK (rewards: {rewards[:3]})")
    
    # Test factory function
    for task_name in ['antmaze_goal_2.0_4.0', 'antmaze_directional_1.0_0.0',
                       'antmaze_path_edges', 'exorl_goal_walker',
                       'exorl_velocity_walker_forward', 'kitchen_subtask_0',
                       'kitchen_all']:
        reward_fn = create_eval_reward_function(task_name, state_dim=29, device=device)
        print(f"  Factory {task_name}: OK (type: {type(reward_fn).__name__})")
    
    # Test get_eval_tasks
    for domain in ['antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen']:
        tasks = get_eval_tasks(domain)
        print(f"  {domain}: {len(tasks)} tasks")
    
    print("All evaluation reward function tests passed!")


if __name__ == '__main__':
    test_eval_rewards()