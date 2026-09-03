"""
FRE Zero-Shot Evaluator
========================
Evaluates a trained FRE agent on downstream tasks by:
1. Sampling K encoding states from the offline dataset
2. Computing rewards using the task's reward function
3. Encoding to latent z via the frozen FRE encoder
4. Running the policy in the environment for N episodes
5. Computing normalized returns

Supports AntMaze, ExORL (Walker/Cheetah), and Kitchen domains.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from models.fre_encoder import FREEncoder
from models.iql_agent import IQLAgent
from data.replay_buffer import ReplayBuffer
from evaluation.metrics import (
    EvaluationTask,
    EvaluationResult,
    normalize_returns,
    compute_normalized_score,
    get_domain_normalization,
    make_antmaze_goal_reaching_reward,
    make_antmaze_directional_reward,
    make_antmaze_random_simplex_reward,
    make_antmaze_path_reward,
    make_exorl_goal_reaching_reward,
    make_exorl_velocity_reward,
    make_kitchen_subtask_reward,
    DOMAIN_NORMALIZATION,
)
from utils.helpers import set_seed, to_tensor, to_numpy, get_device

logger = logging.getLogger(__name__)


class FREEvaluator:
    """
    Zero-shot evaluator for FRE agents.

    Given a trained encoder and IQL agent, evaluates on a set of downstream
    tasks by encoding the task's reward function from a few example states
    and then rolling out the conditioned policy in the environment.

    Parameters
    ----------
    encoder : FREEncoder
        Trained (frozen) FRE encoder.
    agent : IQLAgent
        Trained IQL agent.
    replay_buffer : ReplayBuffer
        Replay buffer containing the offline dataset (for sampling encoding states).
    device : torch.device, optional
        Device for tensor computations.
    K_enc : int, default=32
        Number of encoding states to use for reward function encoding.
    deterministic_policy : bool, default=True
        Whether to use deterministic (mean) actions during evaluation.
    """

    def __init__(
        self,
        encoder: FREEncoder,
        agent: IQLAgent,
        replay_buffer: ReplayBuffer,
        device: Optional[torch.device] = None,
        K_enc: int = 32,
        deterministic_policy: bool = True,
    ):
        self.encoder = encoder
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.device = device or get_device()
        self.K_enc = K_enc
        self.deterministic_policy = deterministic_policy

        # Move models to device and set to eval mode
        self.encoder.to(self.device)
        self.agent.to(self.device)
        self.encoder.eval()
        self.agent.eval()

        # Freeze encoder (should already be frozen, but ensure)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_reward_function(
        self,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        encoding_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode a reward function into a latent vector z.

        Parameters
        ----------
        reward_fn : callable
            Reward function that maps states (np.ndarray) to scalar rewards.
        encoding_states : np.ndarray, optional
            Pre-specified encoding states. If None, sampled from replay buffer.

        Returns
        -------
        z : np.ndarray of shape (latent_dim,)
            Latent encoding of the reward function.
        """
        # Sample encoding states if not provided
        if encoding_states is None:
            encoding_states = self.replay_buffer.sample_states(self.K_enc)

        # Ensure we have exactly K_enc states
        if len(encoding_states) < self.K_enc:
            # Pad by repeating
            repeats = (self.K_enc + len(encoding_states) - 1) // len(encoding_states)
            encoding_states = np.tile(encoding_states, (repeats, 1))[:self.K_enc]
        elif len(encoding_states) > self.K_enc:
            encoding_states = encoding_states[:self.K_enc]

        # Compute rewards on encoding states
        rewards = reward_fn(encoding_states)
        if rewards.ndim == 0:
            rewards = np.full(len(encoding_states), rewards)
        rewards = rewards.reshape(-1)

        # Convert to tensors
        states_tensor = to_tensor(encoding_states, device=self.device, dtype=torch.float32)
        rewards_tensor = to_tensor(rewards, device=self.device, dtype=torch.float32)

        # Encode deterministically (use mean, no sampling)
        with torch.no_grad():
            z = self.encoder.encode_deterministic(states_tensor, rewards_tensor)

        return to_numpy(z)

    def evaluate_on_task(
        self,
        task: EvaluationTask,
        env: Any,
        num_episodes: int = 20,
        encoding_states: Optional[np.ndarray] = None,
        max_episode_steps: int = 1000,
        render: bool = False,
        verbose: bool = False,
    ) -> List[float]:
        """
        Evaluate the agent on a single task.

        Parameters
        ----------
        task : EvaluationTask
            The evaluation task (contains reward function and metadata).
        env : gym.Env
            The environment instance.
        num_episodes : int, default=20
            Number of evaluation episodes.
        encoding_states : np.ndarray, optional
            Pre-specified encoding states for the reward function.
        max_episode_steps : int, default=1000
            Maximum steps per episode.
        render : bool, default=False
            Whether to render the environment.
        verbose : bool, default=False
            Whether to log per-episode returns.

        Returns
        -------
        episode_returns : List[float]
            Undiscounted returns for each episode.
        """
        # Encode the task's reward function
        z = self.encode_reward_function(task.reward_fn, encoding_states)
        z_tensor = to_tensor(z, device=self.device, dtype=torch.float32)

        episode_returns = []

        for ep in range(num_episodes):
            state, _ = env.reset()
            # Normalize state if replay buffer has normalization stats
            state = self.replay_buffer.normalize_states(state.reshape(1, -1)).flatten()

            episode_return = 0.0
            done = False
            truncated = False
            step = 0

            while not done and not truncated and step < max_episode_steps:
                # Get action from policy
                state_tensor = to_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = self.agent.get_action(
                        state_tensor,
                        z_tensor.unsqueeze(0),
                        deterministic=self.deterministic_policy,
                    )

                # Step environment
                next_state, reward, done, truncated, info = env.step(action)
                next_state = self.replay_buffer.normalize_states(
                    next_state.reshape(1, -1)
                ).flatten()

                episode_return += reward
                state = next_state
                step += 1

                if render:
                    env.render()

            episode_returns.append(episode_return)

            if verbose:
                logger.info(f"  Episode {ep + 1}/{num_episodes}: return = {episode_return:.2f}")

        return episode_returns

    def evaluate_all_tasks(
        self,
        tasks: List[EvaluationTask],
        env_factory: Callable[[], Any],
        num_episodes: int = 20,
        encoding_states_per_task: Optional[Dict[str, np.ndarray]] = None,
        max_episode_steps: int = 1000,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate the agent on all given tasks.

        Parameters
        ----------
        tasks : List[EvaluationTask]
            List of evaluation tasks.
        env_factory : callable
            Function that creates a fresh environment instance.
        num_episodes : int, default=20
            Number of episodes per task.
        encoding_states_per_task : dict, optional
            Pre-specified encoding states keyed by task name.
        max_episode_steps : int, default=1000
            Maximum steps per episode.
        verbose : bool, default=True
            Whether to log progress.

        Returns
        -------
        result : EvaluationResult
            Aggregated evaluation results.
        """
        result = EvaluationResult()

        for task in tasks:
            if verbose:
                logger.info(f"Evaluating task: {task.name} ({task.description})")

            # Get encoding states for this task if provided
            enc_states = None
            if encoding_states_per_task is not None:
                enc_states = encoding_states_per_task.get(task.name, None)

            # Create fresh environment
            env = env_factory()

            try:
                episode_returns = self.evaluate_on_task(
                    task=task,
                    env=env,
                    num_episodes=num_episodes,
                    encoding_states=enc_states,
                    max_episode_steps=max_episode_steps,
                    verbose=verbose,
                )

                result.add_result(task.name, episode_returns)

                if verbose:
                    mean_ret = np.mean(episode_returns)
                    std_ret = np.std(episode_returns)
                    logger.info(f"  Task '{task.name}': mean return = {mean_ret:.2f} ± {std_ret:.2f}")

            finally:
                env.close()

        return result

    def evaluate_with_normalization(
        self,
        tasks: List[EvaluationTask],
        env_factory: Callable[[], Any],
        domain: str,
        num_episodes: int = 20,
        encoding_states_per_task: Optional[Dict[str, np.ndarray]] = None,
        max_episode_steps: int = 1000,
        verbose: bool = True,
    ) -> Tuple[EvaluationResult, Dict[str, Tuple[float, float]]]:
        """
        Evaluate and return both raw and normalized results.

        Parameters
        ----------
        tasks : List[EvaluationTask]
            List of evaluation tasks.
        env_factory : callable
            Function that creates a fresh environment instance.
        domain : str
            Domain name for normalization ('antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen').
        num_episodes : int, default=20
            Number of episodes per task.
        encoding_states_per_task : dict, optional
            Pre-specified encoding states keyed by task name.
        max_episode_steps : int, default=1000
            Maximum steps per episode.
        verbose : bool, default=True
            Whether to log progress.

        Returns
        -------
        raw_result : EvaluationResult
            Raw (unnormalized) evaluation results.
        normalized_stats : Dict[str, Tuple[float, float]]
            Task name -> (mean_normalized_return, std_normalized_return).
        """
        raw_result = self.evaluate_all_tasks(
            tasks=tasks,
            env_factory=env_factory,
            num_episodes=num_episodes,
            encoding_states_per_task=encoding_states_per_task,
            max_episode_steps=max_episode_steps,
            verbose=verbose,
        )

        # Get normalization bounds
        min_return, max_return = get_domain_normalization(domain)

        # Compute normalized stats
        normalized_stats = {}
        for task_name, (mean_raw, std_raw) in raw_result.get_all_task_stats().items():
            # Normalize the mean
            norm_mean = normalize_returns(
                np.array([mean_raw]), min_return, max_return
            )[0]
            # Approximate normalized std
            norm_std = (std_raw / (max_return - min_return + 1e-8)) * 100.0
            normalized_stats[task_name] = (norm_mean, norm_std)

        if verbose:
            logger.info(f"\nNormalized results (domain={domain}):")
            for task_name, (mean_norm, std_norm) in normalized_stats.items():
                logger.info(f"  {task_name}: {mean_norm:.1f} ± {std_norm:.1f}")
            # Overall average
            overall_mean = np.mean([m for m, _ in normalized_stats.values()])
            overall_std = np.mean([s for _, s in normalized_stats.values()])
            logger.info(f"  Overall average: {overall_mean:.1f} ± {overall_std:.1f}")

        return raw_result, normalized_stats


# ---------------------------------------------------------------------------
# Domain-Specific Task Builders
# ---------------------------------------------------------------------------

def build_antmaze_tasks(
    state_dim: int = 29,
    num_goals: int = 5,
    num_directions: int = 5,
    num_simplex: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvaluationTask]:
    """
    Build the standard AntMaze evaluation tasks as described in the paper.

    Task categories:
    - Goal-reaching: random goals
    - Directional: move in (x, y) direction
    - Random-simplex: procedural noise / random Fourier features
    - Path tasks: loop, edges, center

    Parameters
    ----------
    state_dim : int, default=29
        State dimension of the AntMaze environment.
    num_goals : int, default=5
        Number of goal-reaching tasks.
    num_directions : int, default=5
        Number of directional tasks.
    num_simplex : int, default=5
        Number of random-simplex tasks.
    rng : np.random.RandomState, optional
        Random state for reproducibility.

    Returns
    -------
    tasks : List[EvaluationTask]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = []

    # --- Goal-reaching tasks ---
    # Goals are (x, y) positions; state[:2] typically holds position in AntMaze
    for i in range(num_goals):
        goal = rng.uniform(-2, 20, size=2)  # AntMaze coordinates roughly in this range
        reward_fn = make_antmaze_goal_reaching_reward(goal, threshold=0.5)
        tasks.append(
            EvaluationTask(
                name=f"ant-goal-{i}",
                reward_fn=reward_fn,
                description=f"Goal-reaching to ({goal[0]:.1f}, {goal[1]:.1f})",
            )
        )

    # --- Directional tasks ---
    for i in range(num_directions):
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        reward_fn = make_antmaze_directional_reward(direction)
        tasks.append(
            EvaluationTask(
                name=f"ant-directional-{i}",
                reward_fn=reward_fn,
                description=f"Directional ({direction[0]:.2f}, {direction[1]:.2f})",
            )
        )

    # --- Random-simplex tasks ---
    for i in range(num_simplex):
        reward_fn = make_antmaze_random_simplex_reward(state_dim, num_frequencies=10, rng=rng)
        tasks.append(
            EvaluationTask(
                name=f"ant-simplex-{i}",
                reward_fn=reward_fn,
                description=f"Random simplex function {i}",
            )
        )

    # --- Path tasks ---
    # Loop path
    t = np.linspace(0, 2 * np.pi, 50)
    loop_path = np.stack([5 * np.cos(t) + 8, 5 * np.sin(t) + 8], axis=1)
    tasks.append(
        EvaluationTask(
            name="ant-path-loop",
            reward_fn=make_antmaze_path_reward(loop_path, threshold=0.5),
            description="Path: loop",
        )
    )

    # Edges path
    edges_path = np.array([
        [0, 0], [18, 0], [18, 18], [0, 18], [0, 0]
    ])
    tasks.append(
        EvaluationTask(
            name="ant-path-edges",
            reward_fn=make_antmaze_path_reward(edges_path, threshold=0.5),
            description="Path: edges",
        )
    )

    # Center path
    center_path = np.array([
        [4, 4], [14, 4], [14, 14], [4, 14]
    ])
    tasks.append(
        EvaluationTask(
            name="ant-path-center",
            reward_fn=make_antmaze_path_reward(center_path, threshold=0.5),
            description="Path: center",
        )
    )

    return tasks


def build_exorl_walker_tasks(
    num_goals: int = 5,
    num_velocities: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvaluationTask]:
    """
    Build ExORL Walker evaluation tasks.

    Parameters
    ----------
    num_goals : int, default=5
        Number of goal-reaching tasks.
    num_velocities : int, default=5
        Number of velocity tasks.
    rng : np.random.RandomState, optional

    Returns
    -------
    tasks : List[EvaluationTask]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = []

    # Goal-reaching tasks
    for i in range(num_goals):
        goal = rng.uniform(-1, 1, size=8)  # Walker state dim ~8
        reward_fn = make_exorl_goal_reaching_reward(goal, threshold=0.5)
        tasks.append(
            EvaluationTask(
                name=f"walker-goal-{i}",
                reward_fn=reward_fn,
                description=f"Goal-reaching {i}",
            )
        )

    # Velocity tasks
    for i in range(num_velocities):
        target_vel = rng.uniform(0.5, 3.0)
        reward_fn = make_exorl_velocity_reward(target_vel, velocity_idx=0)
        tasks.append(
            EvaluationTask(
                name=f"walker-velocity-{i}",
                reward_fn=reward_fn,
                description=f"Velocity {target_vel:.1f}",
            )
        )

    return tasks


def build_exorl_cheetah_tasks(
    num_goals: int = 5,
    num_velocities: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvaluationTask]:
    """
    Build ExORL Cheetah evaluation tasks.

    Parameters
    ----------
    num_goals : int, default=5
        Number of goal-reaching tasks.
    num_velocities : int, default=5
        Number of velocity tasks.
    rng : np.random.RandomState, optional

    Returns
    -------
    tasks : List[EvaluationTask]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    tasks = []

    # Goal-reaching tasks
    for i in range(num_goals):
        goal = rng.uniform(-1, 1, size=17)  # Cheetah state dim ~17
        reward_fn = make_exorl_goal_reaching_reward(goal, threshold=0.5)
        tasks.append(
            EvaluationTask(
                name=f"cheetah-goal-{i}",
                reward_fn=reward_fn,
                description=f"Goal-reaching {i}",
            )
        )

    # Velocity tasks
    for i in range(num_velocities):
        target_vel = rng.uniform(1.0, 5.0)
        reward_fn = make_exorl_velocity_reward(target_vel, velocity_idx=0)
        tasks.append(
            EvaluationTask(
                name=f"cheetah-velocity-{i}",
                reward_fn=reward_fn,
                description=f"Velocity {target_vel:.1f}",
            )
        )

    return tasks


def build_kitchen_tasks(
    num_subtasks: int = 7,
) -> List[EvaluationTask]:
    """
    Build Kitchen evaluation tasks (7 subtasks).

    The 7 subtasks are:
    0: microwave
    1: kettle
    2: light switch
    3: slide cabinet
    4: hinge cabinet
    5: bottom burner
    6: top burner

    Parameters
    ----------
    num_subtasks : int, default=7
        Number of subtasks to evaluate.

    Returns
    -------
    tasks : List[EvaluationTask]
    """
    subtask_names = [
        "microwave",
        "kettle",
        "light_switch",
        "slide_cabinet",
        "hinge_cabinet",
        "bottom_burner",
        "top_burner",
    ]

    tasks = []
    for i in range(min(num_subtasks, len(subtask_names))):
        reward_fn = make_kitchen_subtask_reward(i)
        tasks.append(
            EvaluationTask(
                name=f"kitchen-{subtask_names[i]}",
                reward_fn=reward_fn,
                description=f"Subtask: {subtask_names[i]}",
            )
        )

    return tasks


def build_tasks_for_domain(
    domain: str,
    state_dim: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[EvaluationTask]:
    """
    Build evaluation tasks for a given domain.

    Parameters
    ----------
    domain : str
        One of 'antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen'.
    state_dim : int, optional
        State dimension (auto-detected if not provided).
    rng : np.random.RandomState, optional

    Returns
    -------
    tasks : List[EvaluationTask]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    domain_lower = domain.lower()

    if domain_lower == "antmaze":
        return build_antmaze_tasks(
            state_dim=state_dim or 29,
            rng=rng,
        )
    elif domain_lower == "exorl_walker":
        return build_exorl_walker_tasks(rng=rng)
    elif domain_lower == "exorl_cheetah":
        return build_exorl_cheetah_tasks(rng=rng)
    elif domain_lower == "kitchen":
        return build_kitchen_tasks()
    else:
        raise ValueError(f"Unknown domain: {domain}. "
                         f"Expected one of: antmaze, exorl_walker, exorl_cheetah, kitchen.")


# ---------------------------------------------------------------------------
# Multi-Seed Evaluation Runner
# ---------------------------------------------------------------------------

def run_multi_seed_evaluation(
    checkpoint_paths: List[str],
    domain: str,
    env_name: str,
    replay_buffer: ReplayBuffer,
    num_episodes: int = 20,
    K_enc: int = 32,
    device: Optional[torch.device] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
    verbose: bool = True,
) -> Tuple[EvaluationResult, Dict[str, Tuple[float, float]]]:
    """
    Run evaluation across multiple training seeds.

    Parameters
    ----------
    checkpoint_paths : List[str]
        Paths to model checkpoints for each seed.
    domain : str
        Domain name.
    env_name : str
        Gym environment name.
    replay_buffer : ReplayBuffer
        Replay buffer for sampling encoding states.
    num_episodes : int, default=20
        Number of episodes per task per seed.
    K_enc : int, default=32
        Number of encoding states.
    device : torch.device, optional
    state_dim : int, optional
    action_dim : int, optional
    rng : np.random.RandomState, optional
    verbose : bool, default=True

    Returns
    -------
    aggregated_result : EvaluationResult
        Results aggregated across all seeds.
    normalized_stats : Dict[str, Tuple[float, float]]
        Mean and std of normalized returns across seeds.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Build tasks
    tasks = build_tasks_for_domain(domain, state_dim=state_dim, rng=rng)

    # Infer dimensions from replay buffer if not provided
    if state_dim is None:
        sample = replay_buffer.sample(1)
        state_dim = sample["states"].shape[1]
    if action_dim is None:
        sample = replay_buffer.sample(1)
        action_dim = sample["actions"].shape[1]

    # Create environment factory
    import gym

    def env_factory():
        return gym.make(env_name)

    # Aggregate results across seeds
    aggregated_result = EvaluationResult()
    all_seed_raw_results = []

    for seed_idx, ckpt_path in enumerate(checkpoint_paths):
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating seed {seed_idx + 1}/{len(checkpoint_paths)}")
            logger.info(f"Checkpoint: {ckpt_path}")
            logger.info(f"{'='*60}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device or get_device())

        # Reconstruct models
        from models.fre_encoder import FREEncoder
        from models.iql_agent import IQLAgent

        encoder = FREEncoder(
            state_dim=state_dim,
            embed_dim=checkpoint.get("embed_dim", 256),
            latent_dim=checkpoint.get("latent_dim", 64),
        )
        agent = IQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=checkpoint.get("latent_dim", 64),
        )

        # Load weights
        if "encoder" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder"])
        elif "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])

        if "agent" in checkpoint:
            agent.load_state_dict(checkpoint["agent"])
        elif "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])

        # Create evaluator
        evaluator = FREEvaluator(
            encoder=encoder,
            agent=agent,
            replay_buffer=replay_buffer,
            device=device,
            K_enc=K_enc,
            deterministic_policy=True,
        )

        # Evaluate
        raw_result, _ = evaluator.evaluate_with_normalization(
            tasks=tasks,
            env_factory=env_factory,
            domain=domain,
            num_episodes=num_episodes,
            verbose=verbose,
        )

        # Aggregate
        for task_name, (mean_raw, std_raw) in raw_result.get_all_task_stats().items():
            # Store raw episode returns for this seed
            # We need to get the raw episode returns from the result
            # The EvaluationResult stores per-seed data
            aggregated_result.add_result(task_name, raw_result._raw_returns.get(task_name, [mean_raw]), seed=seed_idx)

        all_seed_raw_results.append(raw_result)

    # Compute normalized stats across seeds
    min_return, max_return = get_domain_normalization(domain)
    normalized_stats = {}

    for task in tasks:
        task_name = task.name
        seed_means = []
        for seed_idx in range(len(checkpoint_paths)):
            seed_data = aggregated_result._raw_returns.get(task_name, [])
            if seed_idx < len(seed_data):
                seed_means.append(np.mean(seed_data[seed_idx]) if isinstance(seed_data[seed_idx], list) else seed_data[seed_idx])

        if seed_means:
            norm_values = normalize_returns(np.array(seed_means), min_return, max_return)
            normalized_stats[task_name] = (float(np.mean(norm_values)), float(np.std(norm_values)))

    if verbose:
        logger.info(f"\nFinal aggregated results ({len(checkpoint_paths)} seeds):")
        for task_name, (mean_norm, std_norm) in normalized_stats.items():
            logger.info(f"  {task_name}: {mean_norm:.1f} ± {std_norm:.1f}")
        overall_mean = np.mean([m for m, _ in normalized_stats.values()])
        overall_std = np.mean([s for _, s in normalized_stats.values()])
        logger.info(f"  Overall average: {overall_mean:.1f} ± {overall_std:.1f}")

    return aggregated_result, normalized_stats


# ---------------------------------------------------------------------------
# Convenience function for single-seed evaluation from a trainer
# ---------------------------------------------------------------------------

def evaluate_from_trainer(
    trainer,  # FRETrainer instance
    domain: str,
    env_name: str,
    num_episodes: int = 20,
    K_enc: int = 32,
    verbose: bool = True,
) -> Tuple[EvaluationResult, Dict[str, Tuple[float, float]]]:
    """
    Evaluate a trained FRETrainer on downstream tasks.

    Parameters
    ----------
    trainer : FRETrainer
        Trained trainer instance (contains encoder, agent, replay_buffer).
    domain : str
        Domain name.
    env_name : str
        Gym environment name.
    num_episodes : int, default=20
        Number of episodes per task.
    K_enc : int, default=32
        Number of encoding states.
    verbose : bool, default=True

    Returns
    -------
    raw_result : EvaluationResult
    normalized_stats : Dict[str, Tuple[float, float]]
    """
    import gym

    # Get components from trainer
    encoder = trainer.get_encoder()
    agent = trainer.get_agent()
    replay_buffer = trainer.replay_buffer
    device = trainer.device

    # Build tasks
    state_dim = replay_buffer.sample(1)["states"].shape[1]
    tasks = build_tasks_for_domain(domain, state_dim=state_dim)

    # Create evaluator
    evaluator = FREEvaluator(
        encoder=encoder,
        agent=agent,
        replay_buffer=replay_buffer,
        device=device,
        K_enc=K_enc,
        deterministic_policy=True,
    )

    def env_factory():
        return gym.make(env_name)

    return evaluator.evaluate_with_normalization(
        tasks=tasks,
        env_factory=env_factory,
        domain=domain,
        num_episodes=num_episodes,
        verbose=verbose,
    )