"""
Co-Evolution Edge Case Test Suite
Tests for Red/Blue adversarial RL edge cases and failure modes

Author: @sarowarzahan414
Date: 2025-11-18
Usage: python test_coevolution_edge_cases.py
       pytest test_coevolution_edge_cases.py -v
"""

import unittest
import numpy as np
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


# Mock environment for testing
class MockCoEvolutionEnv:
    """Mock environment for testing edge cases"""

    def __init__(self, agent_role="red"):
        self.agent_role = agent_role
        self.current_step = 0
        self.max_steps = 50
        self.network_health = 0.7
        self.kill_chain_phase = 0
        self.action_history = []
        self.reward_history = []

    def reset(self):
        self.current_step = 0
        self.network_health = 0.7
        self.kill_chain_phase = 0
        self.action_history = []
        self.reward_history = []
        return np.random.random(20)

    def step(self, action):
        self.current_step += 1
        self.action_history.append(action)

        reward = np.random.uniform(-10, 20)
        self.reward_history.append(reward)

        terminated = self.current_step >= self.max_steps
        return np.random.random(20), reward, terminated, False, {}


# ============================================================================
# Edge Case Detection Functions
# ============================================================================

def calculate_entropy(actions: List[int], num_actions: int = 25) -> float:
    """Calculate Shannon entropy of action distribution"""
    if len(actions) == 0:
        return 0.0

    action_counts = np.bincount(actions, minlength=num_actions)
    action_probs = action_counts / len(actions)

    # Remove zero probabilities
    action_probs = action_probs[action_probs > 0]

    entropy = -np.sum(action_probs * np.log(action_probs))
    max_entropy = np.log(num_actions)

    return entropy / max_entropy if max_entropy > 0 else 0.0


def detect_cyclic_dominance(win_history: List[Dict], window: int = 5) -> Tuple[bool, str]:
    """Detect cyclic win patterns"""
    if len(win_history) < window * 2:
        return False, "Insufficient history"

    wins_red = [h['red_wins'] for h in win_history[-window*2:]]

    first_half = wins_red[:window]
    second_half = wins_red[window:]

    correlation = np.corrcoef(first_half, second_half)[0, 1]

    if abs(correlation) > 0.8:
        return True, f"Cyclic pattern detected (corr={correlation:.2f})"

    return False, "No cycling detected"


def detect_strategy_collapse(action_history: List[int], threshold: float = 0.2) -> Tuple[bool, str]:
    """Detect if agent is using limited strategy diversity"""
    if len(action_history) < 50:
        return False, "Insufficient history"

    entropy = calculate_entropy(action_history[-100:])

    if entropy < threshold:
        return True, f"Low diversity (entropy={entropy:.3f})"

    return False, "Sufficient diversity"


def detect_catastrophic_forgetting(performance_history: List[float]) -> Tuple[bool, str]:
    """Detect if performance against old opponents degrades"""
    if len(performance_history) < 10:
        return False, "Insufficient history"

    old_perf = np.mean(performance_history[:5])
    recent_perf = np.mean(performance_history[-5:])

    if recent_perf < old_perf - 0.2:
        return True, f"Forgetting detected ({old_perf:.1%} → {recent_perf:.1%})"

    return False, "No forgetting"


def detect_runaway_optimization(metrics: Dict, cap: float = 0.95) -> Tuple[bool, List[str]]:
    """Detect if agents exceed realistic capabilities"""
    issues = []

    if metrics.get('red_success', 0) > cap:
        issues.append(f"Red too powerful: {metrics['red_success']:.1%}")

    if metrics.get('blue_defense', 0) > cap:
        issues.append(f"Blue too powerful: {metrics['blue_defense']:.1%}")

    return len(issues) > 0, issues


def detect_deadlock(state_history: List[float], tolerance: float = 0.01) -> Tuple[bool, str]:
    """Detect if game state is not changing"""
    if len(state_history) < 20:
        return False, "Insufficient history"

    recent_states = state_history[-20:]
    variance = np.var(recent_states)

    if variance < tolerance:
        return True, f"State frozen (var={variance:.4f})"

    return False, "State changing normally"


# ============================================================================
# Test Cases
# ============================================================================

class TestCyclicDominance(unittest.TestCase):
    """Test detection of cyclic dominance patterns"""

    def test_detect_cycling(self):
        """Test that cycling is detected"""
        # Create oscillating pattern
        win_history = [
            {'red_wins': 8, 'blue_wins': 2},
            {'red_wins': 2, 'blue_wins': 8},
            {'red_wins': 8, 'blue_wins': 2},
            {'red_wins': 2, 'blue_wins': 8},
            {'red_wins': 8, 'blue_wins': 2},
            {'red_wins': 2, 'blue_wins': 8},
            {'red_wins': 8, 'blue_wins': 2},
            {'red_wins': 2, 'blue_wins': 8},
            {'red_wins': 8, 'blue_wins': 2},
            {'red_wins': 2, 'blue_wins': 8},
        ]

        is_cycling, msg = detect_cyclic_dominance(win_history)
        self.assertTrue(is_cycling, "Should detect cycling pattern")

    def test_no_cycling_detected(self):
        """Test that stable convergence is not flagged as cycling"""
        win_history = [
            {'red_wins': i, 'blue_wins': 10-i}
            for i in range(5, 10)  # Gradual improvement
        ]

        is_cycling, msg = detect_cyclic_dominance(win_history)
        self.assertFalse(is_cycling, "Should not detect cycling in convergence")


class TestStrategyCollapse(unittest.TestCase):
    """Test detection of strategy collapse"""

    def test_detect_single_action(self):
        """Test detection of agent using single action"""
        # Agent uses action 5 repeatedly
        action_history = [5] * 100

        is_collapsed, msg = detect_strategy_collapse(action_history)
        self.assertTrue(is_collapsed, "Should detect single-action strategy")

    def test_detect_low_diversity(self):
        """Test detection of low action diversity"""
        # Agent uses only 2 actions
        action_history = [0, 1] * 50

        is_collapsed, msg = detect_strategy_collapse(action_history)
        self.assertTrue(is_collapsed, "Should detect low diversity")

    def test_no_collapse_with_diversity(self):
        """Test that diverse strategies are not flagged"""
        # Agent uses many different actions
        action_history = [i % 25 for i in range(100)]

        is_collapsed, msg = detect_strategy_collapse(action_history)
        self.assertFalse(is_collapsed, "Should not flag diverse strategy")


class TestCatastrophicForgetting(unittest.TestCase):
    """Test detection of catastrophic forgetting"""

    def test_detect_forgetting(self):
        """Test detection of performance degradation"""
        # Performance decreases over time
        performance_history = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]

        is_forgetting, msg = detect_catastrophic_forgetting(performance_history)
        self.assertTrue(is_forgetting, "Should detect forgetting")

    def test_no_forgetting_with_improvement(self):
        """Test that improving performance is not flagged"""
        performance_history = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

        is_forgetting, msg = detect_catastrophic_forgetting(performance_history)
        self.assertFalse(is_forgetting, "Should not flag improvement")


class TestRunawayOptimization(unittest.TestCase):
    """Test detection of unrealistic agent capabilities"""

    def test_detect_red_too_powerful(self):
        """Test detection of overpowered red agent"""
        metrics = {
            'red_success': 0.98,
            'blue_defense': 0.50
        }

        is_runaway, issues = detect_runaway_optimization(metrics)
        self.assertTrue(is_runaway, "Should detect overpowered red")
        self.assertTrue(any('Red too powerful' in issue for issue in issues))

    def test_detect_blue_too_powerful(self):
        """Test detection of overpowered blue agent"""
        metrics = {
            'red_success': 0.50,
            'blue_defense': 0.97
        }

        is_runaway, issues = detect_runaway_optimization(metrics)
        self.assertTrue(is_runaway, "Should detect overpowered blue")
        self.assertTrue(any('Blue too powerful' in issue for issue in issues))

    def test_balanced_agents_ok(self):
        """Test that balanced agents are not flagged"""
        metrics = {
            'red_success': 0.60,
            'blue_defense': 0.65
        }

        is_runaway, issues = detect_runaway_optimization(metrics)
        self.assertFalse(is_runaway, "Should not flag balanced agents")


class TestDeadlock(unittest.TestCase):
    """Test detection of adversarial deadlock"""

    def test_detect_frozen_state(self):
        """Test detection of unchanging game state"""
        # State stuck at 0.5
        state_history = [0.5] * 25

        is_deadlock, msg = detect_deadlock(state_history)
        self.assertTrue(is_deadlock, "Should detect frozen state")

    def test_normal_variation_ok(self):
        """Test that normal state changes are not flagged"""
        # State varies normally
        state_history = [0.5 + 0.1 * np.sin(i * 0.1) for i in range(25)]

        is_deadlock, msg = detect_deadlock(state_history)
        self.assertFalse(is_deadlock, "Should not flag normal variation")


class TestActionValidation(unittest.TestCase):
    """Test action validation and phase enforcement"""

    def test_phase_requirement_enforcement(self):
        """Test that actions require appropriate phase"""

        def validate_action(action: int, current_phase: int) -> Tuple[bool, str]:
            # Exploitation tools require phase 3+
            if action in [10, 11, 12, 13, 14]:
                if current_phase < 3:
                    return False, "Exploitation requires phase 3+"
            return True, ""

        # Should fail: exploitation at phase 0
        valid, msg = validate_action(12, current_phase=0)
        self.assertFalse(valid, "Should reject exploitation in early phase")

        # Should succeed: exploitation at phase 3
        valid, msg = validate_action(12, current_phase=3)
        self.assertTrue(valid, "Should allow exploitation in correct phase")

    def test_action_cost_enforcement(self):
        """Test that expensive actions require budget"""

        def can_afford(action: int, budget: int) -> bool:
            costs = {0: 1, 12: 20, 24: 50}
            cost = costs.get(action, 10)
            return budget >= cost

        # Should fail: expensive action with low budget
        self.assertFalse(can_afford(24, budget=10), "Should reject expensive action")

        # Should succeed: cheap action with low budget
        self.assertTrue(can_afford(0, budget=10), "Should allow cheap action")


class TestEntropyCalculation(unittest.TestCase):
    """Test entropy calculation for diversity measurement"""

    def test_zero_entropy_single_action(self):
        """Test that single action has zero entropy"""
        actions = [5] * 100
        entropy = calculate_entropy(actions)
        self.assertAlmostEqual(entropy, 0.0, places=2, msg="Single action should have zero entropy")

    def test_max_entropy_uniform(self):
        """Test that uniform distribution has maximum entropy"""
        # Each action used equally
        actions = []
        for i in range(25):
            actions.extend([i] * 4)

        entropy = calculate_entropy(actions, num_actions=25)
        self.assertGreater(entropy, 0.95, "Uniform distribution should have high entropy")

    def test_medium_entropy(self):
        """Test medium entropy for partially diverse actions"""
        # Use 5 actions equally
        actions = []
        for i in range(5):
            actions.extend([i] * 20)

        entropy = calculate_entropy(actions, num_actions=25)
        self.assertGreater(entropy, 0.3, "Should have medium entropy")
        self.assertLess(entropy, 0.7, "Should not be too high")


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility checks"""

    def test_observation_space_mismatch(self):
        """Test detection of observation space mismatch"""

        class MockModel:
            def __init__(self, obs_shape):
                self.observation_space = type('obj', (object,), {'shape': obs_shape})()

        class MockEnv:
            def __init__(self, obs_shape):
                self.observation_space = type('obj', (object,), {'shape': obs_shape})()

        model = MockModel((20,))
        env = MockEnv((62,))

        # Should detect mismatch
        compatible = model.observation_space.shape == env.observation_space.shape
        self.assertFalse(compatible, "Should detect observation space mismatch")

    def test_compatible_spaces(self):
        """Test that matching spaces are compatible"""

        class MockModel:
            def __init__(self, obs_shape):
                self.observation_space = type('obj', (object,), {'shape': obs_shape})()

        class MockEnv:
            def __init__(self, obs_shape):
                self.observation_space = type('obj', (object,), {'shape': obs_shape})()

        model = MockModel((62,))
        env = MockEnv((62,))

        compatible = model.observation_space.shape == env.observation_space.shape
        self.assertTrue(compatible, "Should recognize compatible spaces")


class TestRewardPatterns(unittest.TestCase):
    """Test reward pattern analysis"""

    def test_detect_reward_exploitation(self):
        """Test detection of reward hacking patterns"""

        def is_exploiting_termination(episode_lengths: List[int], rewards: List[float]) -> bool:
            # If agent ends episodes early with high reward, might be exploiting
            avg_length = np.mean(episode_lengths)
            avg_reward = np.mean(rewards)

            if avg_length < 10 and avg_reward > 50:
                return True
            return False

        # Normal episodes
        normal_lengths = [40, 45, 50, 48, 52]
        normal_rewards = [30, 35, 40, 32, 38]
        self.assertFalse(is_exploiting_termination(normal_lengths, normal_rewards))

        # Exploiting termination
        exploit_lengths = [5, 6, 5, 7, 6]
        exploit_rewards = [60, 65, 70, 68, 72]
        self.assertTrue(is_exploiting_termination(exploit_lengths, exploit_rewards))


class TestPhaseProgression(unittest.TestCase):
    """Test kill chain phase progression"""

    def test_detect_phase_stagnation(self):
        """Test detection of being stuck in same phase"""

        def is_stuck(phase_history: List[str], window: int = 20) -> bool:
            if len(phase_history) < window:
                return False
            recent = phase_history[-window:]
            return len(set(recent)) == 1

        # Stuck in reconnaissance
        stuck_history = ['RECONNAISSANCE'] * 25
        self.assertTrue(is_stuck(stuck_history), "Should detect stagnation")

        # Normal progression
        normal_history = ['RECONNAISSANCE'] * 5 + ['WEAPONIZATION'] * 5 + ['EXPLOITATION'] * 5
        self.assertFalse(is_stuck(normal_history), "Should not flag progression")

    def test_phase_progression_reward(self):
        """Test reward for advancing through phases"""

        def get_progression_reward(old_phase: int, new_phase: int) -> float:
            if new_phase > old_phase:
                return (new_phase - old_phase) * 25.0
            return -1.0  # Small penalty for staying

        # Advancing should give reward
        advance_reward = get_progression_reward(0, 1)
        self.assertGreater(advance_reward, 0, "Should reward advancement")

        # Staying should give small penalty
        stay_reward = get_progression_reward(2, 2)
        self.assertLess(stay_reward, 0, "Should penalize staying")


# ============================================================================
# Integration Tests
# ============================================================================

class TestEdgeCaseDetectionSystem(unittest.TestCase):
    """Test integrated edge case detection system"""

    def test_multiple_edge_cases(self):
        """Test detection of multiple simultaneous edge cases"""

        # Create problematic training state
        training_state = {
            'generation': 10,
            'red_action_history': [5] * 100,  # Strategy collapse
            'blue_action_history': [10] * 100,  # Strategy collapse
            'win_history': [
                {'red_wins': 8, 'blue_wins': 2},
                {'red_wins': 2, 'blue_wins': 8},
            ] * 5,  # Cycling
            'metrics': {'red_success': 0.98, 'blue_defense': 0.30}  # Runaway
        }

        # Detect issues
        issues = []

        # Check strategy collapse
        red_collapsed, msg = detect_strategy_collapse(training_state['red_action_history'])
        if red_collapsed:
            issues.append(('strategy_collapse', 'red', msg))

        blue_collapsed, msg = detect_strategy_collapse(training_state['blue_action_history'])
        if blue_collapsed:
            issues.append(('strategy_collapse', 'blue', msg))

        # Check cycling
        is_cycling, msg = detect_cyclic_dominance(training_state['win_history'])
        if is_cycling:
            issues.append(('cyclic_dominance', None, msg))

        # Check runaway
        is_runaway, runaway_issues = detect_runaway_optimization(training_state['metrics'])
        if is_runaway:
            for issue in runaway_issues:
                issues.append(('runaway_optimization', None, issue))

        # Should detect all issues
        self.assertGreater(len(issues), 0, "Should detect multiple edge cases")
        print(f"\nDetected {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue[0]}: {issue[2]}")


def run_edge_case_tests():
    """Run all edge case tests"""
    print("\n" + "="*70)
    print("CO-EVOLUTION EDGE CASE TEST SUITE")
    print("="*70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCyclicDominance,
        TestStrategyCollapse,
        TestCatastrophicForgetting,
        TestRunawayOptimization,
        TestDeadlock,
        TestActionValidation,
        TestEntropyCalculation,
        TestModelCompatibility,
        TestRewardPatterns,
        TestPhaseProgression,
        TestEdgeCaseDetectionSystem,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

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

    if result.wasSuccessful():
        print("\n✅ ALL EDGE CASE TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")

    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_edge_case_tests()
    sys.exit(0 if success else 1)
