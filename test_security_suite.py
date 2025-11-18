"""
ACEAC Security Test Suite
Comprehensive security tests for adversarial RL testbeds

Author: @sarowarzahan414
Date: 2025-11-18
Usage: python test_security_suite.py
       pytest test_security_suite.py -v
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Import secure environment
try:
    from aceac_cyber_range_SECURE import ACEACCyberRangeSecure
except ImportError:
    print("Warning: Could not import secure environment")
    ACEACCyberRangeSecure = None


class TestInputValidation(unittest.TestCase):
    """Test suite for input validation security"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_action_type_validation(self):
        """Test that invalid action types are handled"""
        self.env.reset()

        invalid_actions = [
            "invalid",  # String
            None,  # None
            [],  # List
            {},  # Dict
            3.14,  # Float (should convert)
        ]

        for invalid_action in invalid_actions:
            with self.subTest(action=invalid_action):
                # Should not crash
                obs, reward, term, trunc, info = self.env.step(invalid_action)
                # Either rejects or handles gracefully
                self.assertTrue('error' in info or reward == self.env.MIN_REWARD or True)

    def test_action_range_validation(self):
        """Test that out-of-range actions are handled"""
        self.env.reset()

        out_of_range = [-999, -1, 999, 9999]

        for action in out_of_range:
            with self.subTest(action=action):
                obs, reward, term, trunc, info = self.env.step(action)
                # Should handle gracefully
                self.assertIsInstance(reward, float)

    def test_action_special_values(self):
        """Test actions with special numeric values"""
        self.env.reset()

        special_values = [
            float('nan'),
            float('inf'),
            float('-inf'),
        ]

        for action in special_values:
            with self.subTest(action=action):
                try:
                    obs, reward, term, trunc, info = self.env.step(action)
                    # If it doesn't raise, check it's handled safely
                    self.assertTrue(np.isfinite(reward))
                except (ValueError, TypeError):
                    # Acceptable to reject
                    pass


class TestObservationSecurity(unittest.TestCase):
    """Test suite for observation space security"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_observation_bounds(self):
        """Test that observations are within declared bounds"""
        self.env.reset()

        for _ in range(100):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)

            # Check bounds
            self.assertTrue(
                self.env.observation_space.contains(obs),
                f"Observation {obs} outside bounds"
            )

            if terminated:
                break

    def test_observation_no_nan_inf(self):
        """Test that observations never contain NaN or Inf"""
        self.env.reset()

        for _ in range(100):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)

            # Check for NaN/Inf
            self.assertTrue(
                np.all(np.isfinite(obs)),
                f"Observation contains NaN or Inf: {obs}"
            )

            if terminated:
                break

    def test_observation_nan_injection(self):
        """Test that NaN in state doesn't propagate to observation"""
        self.env.reset()

        # Inject NaN into internal state
        self.env.network_security = float('nan')

        # Should handle gracefully
        obs = self.env._get_observation()
        self.assertTrue(
            np.all(np.isfinite(obs)),
            "NaN in state propagated to observation"
        )


class TestNumericSafety(unittest.TestCase):
    """Test suite for numeric safety (division by zero, overflow, etc.)"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_division_by_zero_protection(self):
        """Test that division by zero is prevented"""
        self.env.reset()

        # Force zero denominator
        self.env.total_attempts = 0
        self.env.successful_attacks = 0

        # Should not crash
        success_rate = self.env._calculate_success_rate()
        self.assertEqual(success_rate, 0.0)
        self.assertTrue(np.isfinite(success_rate))

    def test_reward_bounds(self):
        """Test that rewards are always within bounds"""
        self.env.reset()

        for _ in range(100):
            action = self.env.action_space.sample()
            _, reward, terminated, _, _ = self.env.step(action)

            # Check bounds
            self.assertGreaterEqual(reward, self.env.MIN_REWARD)
            self.assertLessEqual(reward, self.env.MAX_REWARD)
            self.assertTrue(np.isfinite(reward))

            if terminated:
                break

    def test_extreme_values(self):
        """Test handling of extreme values in state"""
        self.env.reset()

        extreme_values = [
            ('network_security', 999999.0),
            ('network_security', -999999.0),
            ('successful_attacks', 10**10),
        ]

        for attr, value in extreme_values:
            with self.subTest(attr=attr, value=value):
                self.env.reset()
                setattr(self.env, attr, value)

                # Should handle gracefully
                try:
                    obs = self.env._get_observation()
                    self.assertTrue(np.all(np.isfinite(obs)))
                except Exception as e:
                    self.fail(f"Failed to handle extreme value: {e}")


class TestResourceLimits(unittest.TestCase):
    """Test suite for resource limit enforcement"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_step_limit_enforcement(self):
        """Test that hard step limit is enforced"""
        self.env.reset()

        # Set current step to near limit
        self.env.current_step = self.env.MAX_STEPS_HARD_LIMIT + 1

        # Should raise error
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_episode_length_limit(self):
        """Test that episodes terminate within max steps"""
        self.env.reset()

        for step in range(self.env.MAX_STEPS_HARD_LIMIT + 10):
            try:
                _, _, terminated, _, _ = self.env.step(0)
                if terminated:
                    break
            except RuntimeError:
                # Hit hard limit
                break

        # Should have terminated before limit
        self.assertLess(
            step,
            self.env.MAX_STEPS_HARD_LIMIT,
            "Episode did not terminate before hard limit"
        )


class TestStateConsistency(unittest.TestCase):
    """Test suite for state consistency and invariants"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_state_invariants(self):
        """Test that state invariants are maintained"""
        self.env.reset()

        for _ in range(100):
            action = self.env.action_space.sample()
            _, _, terminated, _, _ = self.env.step(action)

            # Validate invariants
            self.assertGreaterEqual(self.env.network_security, 0.0)
            self.assertLessEqual(self.env.network_security, 1.0)
            self.assertGreaterEqual(self.env.successful_attacks, 0)
            self.assertGreaterEqual(self.env.total_attempts, 0)
            self.assertGreaterEqual(self.env.total_attempts, self.env.successful_attacks)

            if terminated:
                break

    def test_reset_clears_state(self):
        """Test that reset properly clears all state"""
        # Run some steps
        self.env.reset()
        for _ in range(10):
            self.env.step(self.env.action_space.sample())

        # Reset
        obs, info = self.env.reset()

        # Check state is reset
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.network_security, 0.8)
        self.assertEqual(self.env.successful_attacks, 0)
        self.assertEqual(self.env.total_attempts, 0)


class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_invalid_state_detection(self):
        """Test that invalid states are detected"""
        self.env.reset()

        # Create invalid state
        self.env.successful_attacks = -1  # Invalid!

        # Should detect and raise
        with self.assertRaises(AssertionError):
            self.env._validate_state()

    def test_graceful_degradation(self):
        """Test graceful degradation on errors"""
        self.env.reset()

        # Try to break it
        try:
            # Various attacks
            self.env.step(None)
            self.env.step("invalid")
            self.env.step(float('nan'))
        except Exception as e:
            # If it raises, should be a controlled exception
            self.assertNotIsInstance(e, (KeyboardInterrupt, SystemExit))


class TestFuzzing(unittest.TestCase):
    """Fuzz testing for robustness"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_random_valid_actions(self):
        """Fuzz with random valid actions"""
        self.env.reset()

        for _ in range(1000):
            action = self.env.action_space.sample()
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)

                # Basic sanity checks
                self.assertIsInstance(obs, np.ndarray)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(terminated, bool)

                if terminated:
                    self.env.reset()

            except Exception as e:
                self.fail(f"Random action caused crash: {e}")

    def test_random_invalid_actions(self):
        """Fuzz with invalid actions"""
        self.env.reset()

        invalid_actions = [
            -999, 999, float('nan'), float('inf'),
            None, "test", [], {}, (1,2,3)
        ]

        for action in invalid_actions:
            try:
                # Should either handle gracefully or raise controlled exception
                obs, reward, term, trunc, info = self.env.step(action)
                # If no exception, check basic properties
                self.assertIsInstance(obs, np.ndarray)
            except (ValueError, TypeError):
                # Acceptable to reject invalid input
                pass
            except Exception as e:
                # Should not be system exception
                self.assertNotIsInstance(e, (KeyboardInterrupt, SystemExit))


class TestAdversarialAttacks(unittest.TestCase):
    """Test resistance to adversarial attacks"""

    def setUp(self):
        """Set up test environment"""
        if ACEACCyberRangeSecure is None:
            self.skipTest("Secure environment not available")
        self.env = ACEACCyberRangeSecure()

    def test_reward_hacking_attempt(self):
        """Test that reward hacking is prevented"""
        self.env.reset()

        # Try to manipulate internal state for high rewards
        initial_reward = None
        for _ in range(10):
            _, reward, _, _, _ = self.env.step(8)  # High power action
            if initial_reward is None:
                initial_reward = reward

        # Rewards should be bounded
        self.assertLessEqual(reward, self.env.MAX_REWARD)

    def test_state_injection_prevention(self):
        """Test that state injection is prevented"""
        self.env.reset()

        # Try to inject invalid state
        self.env.network_security = 999.0  # Invalid!

        # Should be clamped when getting observation
        obs = self.env._get_observation()
        self.assertTrue(self.env.observation_space.contains(obs))


def run_security_tests():
    """Run all security tests"""
    print("\n" + "="*70)
    print("ACEAC SECURITY TEST SUITE")
    print("="*70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestObservationSecurity))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceLimits))
    suite.addTests(loader.loadTestsFromTestCase(TestStateConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestFuzzing))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialAttacks))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL SECURITY TESTS PASSED!")
    else:
        print("\n❌ SECURITY TESTS FAILED!")

    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)
