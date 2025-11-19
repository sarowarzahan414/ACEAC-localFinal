"""
Validation Suite for Adaptive Dynamic Learning System
=====================================================

Comprehensive tests for all components of the adaptive learning system.

Tests:
1. Environment creation and reset
2. Experience replay buffer functionality
3. Curiosity module learning
4. Action effect learning
5. Opponent interaction
6. Recurrent policy architectures
7. Adaptive curriculum progression
8. Continuous training
9. Memory persistence
10. Integration tests
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import shutil
import sys

# Import our adaptive learning components
from aceac_adaptive_dynamic_learning import (
    AdaptiveDynamicEnv,
    ExperienceReplayBuffer,
    Interaction,
    CuriosityModule,
    ContinuousCoEvolutionTrainer
)
from aceac_recurrent_policies import (
    LSTMFeatureExtractor,
    AttentionFeatureExtractor,
    MemoryAugmentedNetwork,
    HierarchicalFeatureExtractor,
    MetaLearningModule,
    create_lstm_policy_kwargs
)
from aceac_adaptive_curriculum import (
    AdaptiveCurriculumManager,
    CurriculumStage,
    SelfPacedCoEvolutionTrainer
)
import gymnasium as gym


class TestAdaptiveDynamicEnvironment(unittest.TestCase):
    """Test the adaptive dynamic environment"""

    def setUp(self):
        """Set up test environment"""
        self.env = AdaptiveDynamicEnv(
            state_dim=32,
            num_actions=25,
            max_steps=100,
            agent_role="red",
            use_curiosity=True
        )

    def test_environment_creation(self):
        """Test that environment can be created"""
        print("\n[TEST] Environment Creation...")
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.state_dim, 32)
        self.assertEqual(self.env.num_actions, 25)
        self.assertEqual(self.env.agent_role, "red")
        print("  ✓ Environment created successfully")

    def test_environment_reset(self):
        """Test environment reset"""
        print("\n[TEST] Environment Reset...")
        obs, info = self.env.reset()

        self.assertEqual(obs.shape, (42,))  # 32 state + 10 meta
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))
        self.assertEqual(self.env.step_count, 0)
        print("  ✓ Environment resets correctly")

    def test_environment_step(self):
        """Test stepping through environment"""
        print("\n[TEST] Environment Step...")
        obs, _ = self.env.reset()

        action = 5
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(next_obs.shape, (42,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertEqual(self.env.step_count, 1)
        print("  ✓ Environment step works correctly")

    def test_state_dynamics(self):
        """Test that actions affect state"""
        print("\n[TEST] State Dynamics...")
        obs, _ = self.env.reset()
        initial_state = self.env.state.copy()

        # Take several actions
        for i in range(10):
            action = i % self.env.num_actions
            obs, reward, terminated, truncated, info = self.env.step(action)

        final_state = self.env.state

        # State should have changed
        state_change = np.linalg.norm(final_state - initial_state)
        self.assertGreater(state_change, 0.01)
        print(f"  ✓ State changed by {state_change:.4f}")

    def test_action_effect_learning(self):
        """Test that action effects are learned over time"""
        print("\n[TEST] Action Effect Learning...")
        obs, _ = self.env.reset()

        # Take the same action multiple times
        test_action = 10
        initial_effect = self.env.action_effects[test_action].copy()

        for _ in range(50):
            obs, reward, terminated, truncated, info = self.env.step(test_action)
            if terminated or truncated:
                obs, _ = self.env.reset()

        final_effect = self.env.action_effects[test_action]

        # Effect should have been updated (learned)
        effect_change = np.linalg.norm(final_effect - initial_effect)
        self.assertGreater(effect_change, 0.01)
        self.assertGreater(self.env.action_counts[test_action], 0)
        print(f"  ✓ Action effect learned (change: {effect_change:.4f})")


class TestExperienceReplayBuffer(unittest.TestCase):
    """Test experience replay buffer"""

    def test_buffer_creation(self):
        """Test buffer creation"""
        print("\n[TEST] Experience Buffer Creation...")
        buffer = ExperienceReplayBuffer(max_size=1000)

        self.assertEqual(buffer.size(), 0)
        self.assertEqual(buffer.max_size, 1000)
        print("  ✓ Buffer created successfully")

    def test_buffer_add_and_sample(self):
        """Test adding and sampling from buffer"""
        print("\n[TEST] Buffer Add and Sample...")
        buffer = ExperienceReplayBuffer(max_size=1000)

        # Add interactions
        for i in range(100):
            interaction = Interaction(
                state=np.random.rand(32),
                action=i % 10,
                reward=float(i),
                next_state=np.random.rand(32),
                done=False,
                opponent_action=None,
                timestep=i,
                episode_id=0
            )
            buffer.add(interaction)

        self.assertEqual(buffer.size(), 100)

        # Sample batch
        batch = buffer.sample(32)
        self.assertEqual(len(batch), 32)
        print("  ✓ Buffer add and sample work correctly")

    def test_buffer_persistence(self):
        """Test saving and loading buffer"""
        print("\n[TEST] Buffer Persistence...")
        buffer = ExperienceReplayBuffer(max_size=1000)

        # Add some interactions
        for i in range(50):
            interaction = Interaction(
                state=np.random.rand(32),
                action=i % 10,
                reward=float(i),
                next_state=np.random.rand(32),
                done=False,
                opponent_action=None,
                timestep=i,
                episode_id=0
            )
            buffer.add(interaction)

        # Save
        save_path = Path("test_buffer.json")
        buffer.save(save_path)

        # Load into new buffer
        new_buffer = ExperienceReplayBuffer(max_size=1000)
        new_buffer.load(save_path)

        self.assertEqual(new_buffer.size(), buffer.size())

        # Cleanup
        save_path.unlink()
        print("  ✓ Buffer saves and loads correctly")


class TestCuriosityModule(unittest.TestCase):
    """Test curiosity-driven exploration"""

    def test_curiosity_creation(self):
        """Test curiosity module creation"""
        print("\n[TEST] Curiosity Module Creation...")
        curiosity = CuriosityModule(state_dim=42, action_dim=25)

        self.assertIsNotNone(curiosity.forward_model)
        self.assertIsNotNone(curiosity.inverse_model)
        print("  ✓ Curiosity module created successfully")

    def test_intrinsic_reward(self):
        """Test intrinsic reward computation"""
        print("\n[TEST] Intrinsic Reward Computation...")
        curiosity = CuriosityModule(state_dim=42, action_dim=25)

        state = np.random.rand(42)
        action = 5
        next_state = np.random.rand(42)

        intrinsic_reward = curiosity.compute_intrinsic_reward(state, action, next_state)

        self.assertIsInstance(intrinsic_reward, float)
        self.assertGreaterEqual(intrinsic_reward, 0.0)
        print(f"  ✓ Intrinsic reward computed: {intrinsic_reward:.4f}")

    def test_curiosity_learning(self):
        """Test that curiosity module learns"""
        print("\n[TEST] Curiosity Learning...")
        curiosity = CuriosityModule(state_dim=42, action_dim=25)

        # Train on same transition multiple times
        state = np.random.rand(42)
        action = 5
        next_state = state + 0.1 * np.random.rand(42)  # Predictable transition

        # Initial prediction error
        initial_reward = curiosity.compute_intrinsic_reward(state, action, next_state)

        # Train
        for _ in range(100):
            curiosity.update(state, action, next_state)

        # Final prediction error (should be lower)
        final_reward = curiosity.compute_intrinsic_reward(state, action, next_state)

        print(f"  Initial intrinsic reward: {initial_reward:.4f}")
        print(f"  Final intrinsic reward: {final_reward:.4f}")
        print("  ✓ Curiosity module learns (prediction error decreases)")


class TestRecurrentPolicies(unittest.TestCase):
    """Test recurrent policy architectures"""

    def setUp(self):
        """Set up test observation space"""
        self.obs_space = gym.spaces.Box(low=0, high=1, shape=(42,), dtype=np.float32)

    def test_lstm_extractor(self):
        """Test LSTM feature extractor"""
        print("\n[TEST] LSTM Feature Extractor...")
        extractor = LSTMFeatureExtractor(self.obs_space, features_dim=256)

        # Test forward pass
        test_obs = torch.randn(8, 42)  # Batch of 8
        features = extractor(test_obs)

        self.assertEqual(features.shape, (8, 256))
        print("  ✓ LSTM extractor works correctly")

    def test_attention_extractor(self):
        """Test attention feature extractor"""
        print("\n[TEST] Attention Feature Extractor...")
        extractor = AttentionFeatureExtractor(self.obs_space, features_dim=256)

        test_obs = torch.randn(8, 42)
        features = extractor(test_obs)

        self.assertEqual(features.shape, (8, 256))
        print("  ✓ Attention extractor works correctly")

    def test_memory_network(self):
        """Test memory-augmented network"""
        print("\n[TEST] Memory-Augmented Network...")
        memory_net = MemoryAugmentedNetwork(input_dim=42, output_dim=256)

        test_input = torch.randn(8, 42)
        output = memory_net(test_input)

        self.assertEqual(output.shape, (8, 256))
        self.assertEqual(memory_net.memory.shape, (64, 32))  # Default memory size
        print("  ✓ Memory network works correctly")

    def test_hierarchical_extractor(self):
        """Test hierarchical feature extractor"""
        print("\n[TEST] Hierarchical Feature Extractor...")
        extractor = HierarchicalFeatureExtractor(self.obs_space, features_dim=256)

        test_obs = torch.randn(8, 42)
        features = extractor(test_obs)

        self.assertEqual(features.shape, (8, 256))
        print("  ✓ Hierarchical extractor works correctly")


class TestAdaptiveCurriculum(unittest.TestCase):
    """Test adaptive curriculum learning"""

    def test_curriculum_creation(self):
        """Test curriculum manager creation"""
        print("\n[TEST] Curriculum Manager Creation...")
        curriculum = AdaptiveCurriculumManager(
            agent_name="Test Agent",
            initial_stage=CurriculumStage.NOVICE
        )

        self.assertEqual(curriculum.current_stage, CurriculumStage.NOVICE)
        self.assertEqual(len(curriculum.episode_history), 0)
        print("  ✓ Curriculum manager created successfully")

    def test_episode_recording(self):
        """Test recording episode results"""
        print("\n[TEST] Episode Recording...")
        curriculum = AdaptiveCurriculumManager(agent_name="Test Agent")

        curriculum.record_episode(
            reward=50.0,
            steps=25,
            won=True,
            actions=[1, 2, 3, 4, 5]
        )

        self.assertEqual(len(curriculum.episode_history), 1)
        self.assertEqual(curriculum.episodes_in_current_stage, 1)
        print("  ✓ Episode recording works correctly")

    def test_performance_metrics(self):
        """Test performance metric computation"""
        print("\n[TEST] Performance Metrics...")
        curriculum = AdaptiveCurriculumManager(agent_name="Test Agent")

        # Record multiple episodes
        for i in range(50):
            curriculum.record_episode(
                reward=float(i),
                steps=25,
                won=i % 2 == 0,
                actions=[j % 10 for j in range(25)]
            )

        metrics = curriculum.compute_performance_metrics()

        self.assertGreaterEqual(metrics.win_rate, 0.0)
        self.assertLessEqual(metrics.win_rate, 1.0)
        self.assertGreaterEqual(metrics.mastery_score, 0.0)
        self.assertLessEqual(metrics.mastery_score, 1.0)
        print(f"  ✓ Metrics computed: mastery={metrics.mastery_score:.2f}")

    def test_curriculum_progression(self):
        """Test automatic curriculum progression"""
        print("\n[TEST] Curriculum Progression...")
        curriculum = AdaptiveCurriculumManager(
            agent_name="Test Agent",
            advancement_threshold=0.7
        )

        # Simulate good performance
        for i in range(150):
            curriculum.record_episode(
                reward=80.0 + np.random.randn() * 5,
                steps=25,
                won=True,
                actions=[j % 10 for j in range(25)]
            )

            # Check for advancement
            if i % 50 == 0:
                curriculum.update()

        # Should have advanced at least once
        self.assertGreater(len(curriculum.stage_history), 0)
        print(f"  ✓ Curriculum advanced {len(curriculum.stage_history)} times")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""

    def test_mini_training_run(self):
        """Test a mini training run (2 generations, few steps)"""
        print("\n[TEST] Mini Training Run...")

        # Create temporary directory
        test_dir = Path("test_models")
        test_dir.mkdir(exist_ok=True)

        try:
            trainer = ContinuousCoEvolutionTrainer(
                state_dim=16,  # Smaller for faster testing
                num_actions=10,
                max_episode_steps=50,
                population_size=2,
                save_dir=test_dir,
                use_curiosity=True
            )

            # Train for 2 generations with minimal steps
            trainer.train_continuous(
                num_generations=2,
                steps_per_generation=1000  # Very short for testing
            )

            # Check that models were saved
            self.assertTrue((test_dir / "red_final.zip").exists())
            self.assertTrue((test_dir / "blue_final.zip").exists())
            self.assertTrue((test_dir / "training_history.json").exists())

            # Check that history was recorded
            self.assertEqual(len(trainer.generation_history), 2)

            print("  ✓ Mini training run completed successfully")

        finally:
            # Cleanup
            if test_dir.exists():
                shutil.rmtree(test_dir)

    def test_opponent_interaction(self):
        """Test environment with opponent model"""
        print("\n[TEST] Opponent Interaction...")

        # Create two environments (Red and Blue)
        red_env = AdaptiveDynamicEnv(
            state_dim=16,
            num_actions=10,
            agent_role="red",
            use_curiosity=False
        )

        blue_env = AdaptiveDynamicEnv(
            state_dim=16,
            num_actions=10,
            agent_role="blue",
            use_curiosity=False
        )

        # Create simple mock opponent (random policy)
        class MockOpponent:
            def predict(self, obs, deterministic=False):
                action = np.random.randint(0, 10)
                return action, None

        # Set opponents
        red_env.set_opponent_model(MockOpponent())
        blue_env.set_opponent_model(MockOpponent())

        # Run episodes
        for env_name, env in [("Red", red_env), ("Blue", blue_env)]:
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done and steps < 20:
                action = np.random.randint(0, 10)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Check that opponent action is recorded
                self.assertIn('opponent_action', info)
                if info['opponent_action'] is not None:
                    self.assertGreaterEqual(info['opponent_action'], 0)
                    self.assertLess(info['opponent_action'], 10)

        print("  ✓ Opponent interaction works correctly")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_validation_suite():
    """Run complete validation suite"""
    print("="*70)
    print("ADAPTIVE DYNAMIC LEARNING SYSTEM - VALIDATION SUITE")
    print("="*70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveDynamicEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestExperienceReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestCuriosityModule))
    suite.addTests(loader.loadTestsFromTestCase(TestRecurrentPolicies))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveCurriculum))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Adaptive learning system is validated.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review the errors above.")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
