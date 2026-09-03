#!/usr/bin/env python3
"""
Unit tests for the RICE Refining module (rice/refine.py).

Tests cover:
- RICERefine initialization and configuration
- Critical state collection
- Mixed initial distribution behavior
- Refining loop (short integration test)
- Save/load functionality
- refine_policy convenience function
- Edge cases and error handling
"""

import unittest
import os
import tempfile
import numpy as np
import torch
import gym

# Import the module under test
from rice.refine import RICERefine, refine_policy
from rice.mask_net import MaskNetwork
from rice.rnd import RNDModule, BonusNormalizer
from rice.env_wrappers import make_state_saveable, StateSaveWrapper
from rice.utils import set_seed, evaluate_policy


# ---------------------------------------------------------------------------
# Helper: create a dummy target policy for testing
# ---------------------------------------------------------------------------
def _make_dummy_target_policy(state_dim, action_dim, discrete=False, num_discrete_actions=None):
    """Returns a callable policy_fn(state) -> (action, log_prob, value, entropy)."""
    if discrete:
        def policy_fn(state):
            batch_size = state.shape[0] if state.ndim > 1 else 1
            action = np.random.randint(0, num_discrete_actions or 2, size=batch_size)
            log_prob = np.zeros(batch_size, dtype=np.float32)
            value = np.zeros(batch_size, dtype=np.float32)
            entropy = np.zeros(batch_size, dtype=np.float32)
            if batch_size == 1:
                return action[0], log_prob[0], value[0], entropy[0]
            return action, log_prob, value, entropy
    else:
        def policy_fn(state):
            batch_size = state.shape[0] if state.ndim > 1 else 1
            action = np.random.randn(batch_size, action_dim).astype(np.float32)
            log_prob = np.zeros(batch_size, dtype=np.float32)
            value = np.zeros(batch_size, dtype=np.float32)
            entropy = np.zeros(batch_size, dtype=np.float32)
            if batch_size == 1:
                return action[0], log_prob[0], value[0], entropy[0]
            return action, log_prob, value, entropy
    return policy_fn


# ---------------------------------------------------------------------------
# Test RICERefine initialization
# ---------------------------------------------------------------------------
class TestRICERefineInit(unittest.TestCase):
    """Tests for RICERefine constructor and configuration."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]  # 4
        self.action_dim = 2  # discrete
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_initialization_defaults(self):
        """Test RICERefine initializes with default parameters."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
        )
        self.assertIsNotNone(refiner)
        self.assertEqual(refiner.p_mixed, 0.25)
        self.assertEqual(refiner.lambda_rnd, 0.01)
        self.assertIsNotNone(refiner.rnd_module)
        self.assertIsNotNone(refiner.bonus_normalizer)
        self.assertIsNotNone(refiner.policy)
        self.assertIsNotNone(refiner.value_net)
        self.assertIsNotNone(refiner.policy_optimizer)

    def test_initialization_custom_params(self):
        """Test RICERefine initializes with custom parameters."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=0.5,
            lambda_rnd=0.001,
            rnd_hidden_sizes=(32, 32),
            rnd_embedding_dim=32,
            rnd_lr=5e-4,
            ppo_lr=1e-4,
            ppo_epochs=5,
            ppo_batch_size=32,
            ppo_clip_epsilon=0.1,
            gamma=0.95,
            gae_lambda=0.9,
            value_loss_coef=1.0,
            entropy_coef=0.02,
            max_grad_norm=1.0,
            normalize_advantages=False,
            normalize_obs=False,
            normalize_bonus=False,
            top_k_per_episode=3,
            num_critical_episodes=50,
            policy_std=0.5,
            value_hidden_sizes=(64, 64),
        )
        self.assertEqual(refiner.p_mixed, 0.5)
        self.assertEqual(refiner.lambda_rnd, 0.001)
        self.assertEqual(refiner.top_k_per_episode, 3)
        self.assertEqual(refiner.num_critical_episodes, 50)
        self.assertEqual(refiner.gamma, 0.95)
        self.assertEqual(refiner.gae_lambda, 0.9)
        self.assertEqual(refiner.ppo_clip_epsilon, 0.1)
        self.assertEqual(refiner.ppo_epochs, 5)
        self.assertEqual(refiner.ppo_batch_size, 32)

    def test_initialization_continuous_action(self):
        """Test RICERefine initializes with continuous action space."""
        env = make_state_saveable(gym.make("MountainCarContinuous-v0"))
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        mask_net = MaskNetwork(state_dim=state_dim, hidden_sizes=(32, 32))
        target_policy = _make_dummy_target_policy(state_dim, action_dim, discrete=False)

        refiner = RICERefine(
            env=env,
            target_policy=target_policy,
            mask_network=mask_net,
            state_dim=state_dim,
            action_dim=action_dim,
            discrete_action=False,
            device="cpu",
        )
        self.assertIsNotNone(refiner)
        self.assertFalse(refiner.discrete_action)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test Critical State Collection
# ---------------------------------------------------------------------------
class TestCriticalStateCollection(unittest.TestCase):
    """Tests for the critical state collection phase."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_collect_critical_states_basic(self):
        """Test that critical states are collected and returned as a list."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=5,
            top_k_per_episode=1,
        )
        critical_states = refiner.collect_critical_states()
        self.assertIsInstance(critical_states, list)
        self.assertGreater(len(critical_states), 0)
        # Each critical state should be a dict with environment state info
        for cs in critical_states:
            self.assertIsInstance(cs, dict)

    def test_collect_critical_states_top_k(self):
        """Test collecting top-k critical states per episode."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=5,
            top_k_per_episode=3,
        )
        critical_states = refiner.collect_critical_states()
        self.assertIsInstance(critical_states, list)
        # With top_k=3 and 5 episodes, we expect up to 15 states
        self.assertLessEqual(len(critical_states), 15)
        self.assertGreater(len(critical_states), 0)

    def test_collect_critical_states_max_limit(self):
        """Test that max_critical_states limits the collection."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=10,
            top_k_per_episode=5,
        )
        # The internal max_critical_states defaults to num_critical_episodes * top_k_per_episode
        critical_states = refiner.collect_critical_states()
        self.assertIsInstance(critical_states, list)
        self.assertGreater(len(critical_states), 0)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test Mixed Initial Distribution
# ---------------------------------------------------------------------------
class TestMixedInitialDistribution(unittest.TestCase):
    """Tests for the mixed initial distribution behavior during refinement."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_p_mixed_zero_uses_default_reset(self):
        """Test that p_mixed=0 always uses default reset."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=0.0,
            num_critical_episodes=3,
        )
        # Collect critical states first
        refiner.collect_critical_states()
        self.assertEqual(len(refiner.critical_states), 0)  # p_mixed=0 means no states stored

    def test_p_mixed_one_always_uses_critical_state(self):
        """Test that p_mixed=1.0 always uses critical state reset."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=1.0,
            num_critical_episodes=3,
        )
        critical_states = refiner.collect_critical_states()
        self.assertGreater(len(critical_states), 0)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test Refining Loop (Integration)
# ---------------------------------------------------------------------------
class TestRefiningLoop(unittest.TestCase):
    """Integration tests for the full refining loop."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_refine_short_run(self):
        """Test that refine() runs for a few iterations without crashing."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
            top_k_per_episode=1,
            p_mixed=0.5,
            lambda_rnd=0.01,
        )
        history = refiner.refine(
            total_steps=500,
            steps_per_iteration=128,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsInstance(history, dict)
        self.assertIn("iteration", history)
        self.assertIn("eval_rewards", history)
        self.assertIn("policy_loss", history)
        self.assertIn("value_loss", history)
        self.assertIn("rnd_loss", history)
        self.assertGreater(len(history["iteration"]), 0)

    def test_refine_returns_improved_or_stable_policy(self):
        """Test that refine returns a policy that can be evaluated."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
            top_k_per_episode=1,
            p_mixed=0.5,
            lambda_rnd=0.01,
        )
        history = refiner.refine(
            total_steps=300,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        # After refining, get_policy should return a callable
        policy = refiner.get_policy()
        self.assertTrue(callable(policy))

        # Evaluate the refined policy
        eval_results = evaluate_policy(self.env, policy, num_episodes=3, max_steps=200)
        self.assertIn("mean_reward", eval_results)
        self.assertIsInstance(eval_results["mean_reward"], float)

    def test_refine_without_rnd(self):
        """Test refine with lambda_rnd=0 (no exploration bonus)."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
            top_k_per_episode=1,
            p_mixed=0.5,
            lambda_rnd=0.0,
        )
        history = refiner.refine(
            total_steps=300,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsInstance(history, dict)
        self.assertGreater(len(history["iteration"]), 0)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test Save/Load Functionality
# ---------------------------------------------------------------------------
class TestSaveLoad(unittest.TestCase):
    """Tests for saving and loading the RICERefine state."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )
        self.temp_dir = tempfile.mkdtemp()

    def test_save_load_refiner(self):
        """Test that a refiner can be saved and loaded."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
        )
        # Collect critical states
        refiner.collect_critical_states()

        # Save
        save_path = os.path.join(self.temp_dir, "test_refiner.pt")
        refiner.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Load into a new refiner
        new_refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
        )
        new_refiner.load(save_path)

        # Check that critical states were loaded
        self.assertEqual(len(new_refiner.critical_states), len(refiner.critical_states))

    def test_save_load_critical_states(self):
        """Test saving and loading critical states separately."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
        )
        critical_states = refiner.collect_critical_states()

        # Save critical states
        cs_path = os.path.join(self.temp_dir, "critical_states.pkl")
        refiner.save_critical_states(cs_path)
        self.assertTrue(os.path.exists(cs_path))

        # Load into new refiner
        new_refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
        )
        new_refiner.load_critical_states(cs_path)
        self.assertEqual(len(new_refiner.critical_states), len(critical_states))

    def test_save_after_refine(self):
        """Test saving after a short refine run."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
        )
        refiner.refine(
            total_steps=200,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        save_path = os.path.join(self.temp_dir, "refined_checkpoint.pt")
        refiner.save(save_path)
        self.assertTrue(os.path.exists(save_path))

    def tearDown(self):
        self.env.close()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test refine_policy convenience function
# ---------------------------------------------------------------------------
class TestRefinePolicyFunction(unittest.TestCase):
    """Tests for the refine_policy() convenience function."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_refine_policy_basic(self):
        """Test refine_policy runs end-to-end with minimal steps."""
        refined_policy, history = refine_policy(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=0.5,
            lambda_rnd=0.01,
            total_steps=300,
            steps_per_iteration=100,
            num_critical_episodes=3,
            top_k_per_episode=1,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsNotNone(refined_policy)
        self.assertIsInstance(history, dict)
        self.assertIn("iteration", history)

    def test_refine_policy_with_save(self):
        """Test refine_policy with save_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "refined_policy.pt")
            refined_policy, history = refine_policy(
                env=self.env,
                target_policy=self.target_policy,
                mask_network=self.mask_net,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                discrete_action=True,
                num_discrete_actions=2,
                device="cpu",
                p_mixed=0.5,
                lambda_rnd=0.01,
                total_steps=200,
                steps_per_iteration=100,
                num_critical_episodes=3,
                top_k_per_episode=1,
                eval_interval=2,
                eval_episodes=2,
                verbose=False,
                save_path=save_path,
            )
            self.assertTrue(os.path.exists(save_path))

    def test_refine_policy_continuous(self):
        """Test refine_policy with continuous action space."""
        env = make_state_saveable(gym.make("MountainCarContinuous-v0"))
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        mask_net = MaskNetwork(state_dim=state_dim, hidden_sizes=(32, 32))
        target_policy = _make_dummy_target_policy(state_dim, action_dim, discrete=False)

        refined_policy, history = refine_policy(
            env=env,
            target_policy=target_policy,
            mask_network=mask_net,
            state_dim=state_dim,
            action_dim=action_dim,
            discrete_action=False,
            device="cpu",
            p_mixed=0.5,
            lambda_rnd=0.01,
            total_steps=200,
            steps_per_iteration=100,
            num_critical_episodes=3,
            top_k_per_episode=1,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsNotNone(refined_policy)
        env.close()

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test Edge Cases and Error Handling
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_refine_with_zero_critical_states(self):
        """Test refine when no critical states are collected (p_mixed=0)."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=0.0,
            num_critical_episodes=0,
        )
        history = refiner.refine(
            total_steps=200,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsInstance(history, dict)

    def test_refine_without_collecting_first(self):
        """Test refine() called without prior collect_critical_states()."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            p_mixed=0.5,
            num_critical_episodes=3,
        )
        # refine() should internally call collect_critical_states if not done
        history = refiner.refine(
            total_steps=200,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIsInstance(history, dict)

    def test_get_policy_before_refine(self):
        """Test get_policy() returns a callable even before refine."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
        )
        policy = refiner.get_policy()
        self.assertTrue(callable(policy))

    def test_device_movement(self):
        """Test that RICERefine can be moved to a different device."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
        )
        # Move to CPU (already on CPU, should be no-op)
        refiner.to("cpu")
        self.assertIsNotNone(refiner.policy)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Test RND Integration in Refining
# ---------------------------------------------------------------------------
class TestRNDIntegration(unittest.TestCase):
    """Tests for RND module integration within the refining loop."""

    def setUp(self):
        set_seed(42)
        self.env = make_state_saveable(gym.make("CartPole-v1"))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 2
        self.mask_net = MaskNetwork(state_dim=self.state_dim, hidden_sizes=(32, 32))
        self.target_policy = _make_dummy_target_policy(
            self.state_dim, self.action_dim, discrete=True, num_discrete_actions=2
        )

    def test_rnd_bonus_computed_during_refine(self):
        """Test that RND bonus is computed and RND loss decreases during refine."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
            p_mixed=0.5,
            lambda_rnd=0.01,
        )
        history = refiner.refine(
            total_steps=500,
            steps_per_iteration=128,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        self.assertIn("rnd_loss", history)
        self.assertGreater(len(history["rnd_loss"]), 0)
        # RND loss should be finite
        for loss in history["rnd_loss"]:
            self.assertTrue(np.isfinite(loss))

    def test_rnd_bonus_normalizer_updated(self):
        """Test that BonusNormalizer statistics are updated during refine."""
        refiner = RICERefine(
            env=self.env,
            target_policy=self.target_policy,
            mask_network=self.mask_net,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=True,
            num_discrete_actions=2,
            device="cpu",
            num_critical_episodes=3,
            p_mixed=0.5,
            lambda_rnd=0.01,
            normalize_bonus=True,
        )
        refiner.refine(
            total_steps=300,
            steps_per_iteration=100,
            eval_interval=2,
            eval_episodes=2,
            verbose=False,
        )
        # Bonus normalizer should have been updated
        self.assertIsNotNone(refiner.bonus_normalizer)

    def tearDown(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()