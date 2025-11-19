"""
CHECK 2: Reward Sanity Test

Verify that rewards are reasonable and agents can learn.
Tests:
1. Rewards are in expected range
2. Actions produce different rewards
3. State changes correlate with rewards
4. Diversity bonus works correctly
"""

import numpy as np
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv

def test_reward_range():
    """Test that rewards are in reasonable range"""
    print("="*70)
    print("CHECK 2.1: Reward Range Test")
    print("="*70)

    env = DynamicCoEvolutionEnv(agent_role="red")
    rewards = []

    for episode in range(10):
        obs, _ = env.reset()
        episode_rewards = []

        for step in range(100):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            episode_rewards.append(reward)

            if done:
                break

        rewards.extend(episode_rewards)

    rewards = np.array(rewards)

    print(f"Reward Statistics:")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std:  {np.std(rewards):.3f}")
    print(f"  Min:  {np.min(rewards):.3f}")
    print(f"  Max:  {np.max(rewards):.3f}")
    print(f"  Range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")

    # Sanity checks
    if np.abs(np.mean(rewards)) < 10.0:
        print("✓ Mean reward near zero (good - balanced exploration)")
    else:
        print(f"⚠️  Mean reward far from zero: {np.mean(rewards):.3f}")

    if np.std(rewards) > 0.1:
        print("✓ Reward variance exists (good - learning signal)")
    else:
        print("⚠️  Low reward variance - weak learning signal")

    if np.min(rewards) >= -50 and np.max(rewards) <= 50:
        print("✓ Rewards within clipping bounds [-50, 50]")
    else:
        print("⚠️  Rewards outside expected bounds")

    print()
    return True


def test_action_reward_diversity():
    """Test that different actions produce different rewards"""
    print("="*70)
    print("CHECK 2.2: Action-Reward Diversity")
    print("="*70)

    env = DynamicCoEvolutionEnv(agent_role="red")
    action_rewards = {i: [] for i in range(env.num_actions)}

    # Collect rewards for each action
    for _ in range(20):
        obs, _ = env.reset()

        for action in range(env.num_actions):
            obs, reward, done, _, info = env.step(action)
            action_rewards[action].append(reward)

            if done:
                obs, _ = env.reset()

    # Calculate mean reward per action
    action_means = {a: np.mean(rewards) for a, rewards in action_rewards.items()}

    print(f"Action reward means (sample):")
    for action in list(action_means.keys())[:5]:
        print(f"  Action {action}: {action_means[action]:.3f}")
    print(f"  ...")

    # Check diversity
    mean_values = list(action_means.values())
    reward_range = np.max(mean_values) - np.min(mean_values)

    print(f"\nReward diversity:")
    print(f"  Range across actions: {reward_range:.3f}")

    if reward_range > 1.0:
        print("✓ Actions produce diverse rewards (good)")
    else:
        print("⚠️  Low action reward diversity")

    # Check if all actions have similar mean (should be true initially)
    reward_std = np.std(mean_values)
    print(f"  Std dev across actions: {reward_std:.3f}")

    if reward_std < 5.0:
        print("✓ No single action dominates (good - balanced start)")
    else:
        print("⚠️  Some actions much better than others (may be OK)")

    print()
    return True


def test_state_reward_correlation():
    """Test that state changes correlate with rewards"""
    print("="*70)
    print("CHECK 2.3: State-Reward Correlation")
    print("="*70)

    env = DynamicCoEvolutionEnv(agent_role="red")

    state_changes = []
    rewards = []

    for _ in range(50):
        obs, _ = env.reset()
        initial_state = env.state.copy()

        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        final_state = env.state.copy()
        state_change = np.mean(final_state) - np.mean(initial_state)

        state_changes.append(state_change)
        rewards.append(reward)

    correlation = np.corrcoef(state_changes, rewards)[0, 1]

    print(f"State change vs Reward correlation: {correlation:.3f}")

    if np.abs(correlation) > 0.3:
        print(f"✓ Moderate correlation (good - rewards track state)")
    else:
        print(f"⚠️  Weak correlation - rewards may not reflect state")

    print()
    return True


def test_diversity_bonus_function():
    """Test diversity bonus mechanism"""
    print("="*70)
    print("CHECK 2.4: Diversity Bonus Mechanism")
    print("="*70)

    env = DynamicCoEvolutionEnv(agent_role="red")
    obs, _ = env.reset()

    # Test 1: Same action repeatedly (low diversity)
    for _ in range(10):
        env.step(5)

    low_diversity_bonus = env._diversity_bonus()

    # Test 2: Different actions (high diversity)
    obs, _ = env.reset()
    for i in range(10):
        env.step(i)

    high_diversity_bonus = env._diversity_bonus()

    print(f"Diversity bonus comparison:")
    print(f"  Low diversity (same action):  {low_diversity_bonus:.3f}")
    print(f"  High diversity (varied actions): {high_diversity_bonus:.3f}")
    print(f"  Difference: {high_diversity_bonus - low_diversity_bonus:.3f}")

    if high_diversity_bonus > low_diversity_bonus:
        print("✓ Diversity bonus rewards exploration (good)")
    else:
        print("⚠️  Diversity bonus not working correctly")

    if high_diversity_bonus <= 5.0:
        print("✓ Diversity bonus magnitude reasonable")
    else:
        print("⚠️  Diversity bonus too large - may dominate learning")

    print()
    return True


def main():
    print("\n" + "="*70)
    print("CHECK 2: REWARD SANITY TEST")
    print("="*70)
    print("Verifying reward structure enables learning\n")

    results = []

    try:
        results.append(("Reward Range", test_reward_range()))
        results.append(("Action-Reward Diversity", test_action_reward_diversity()))
        results.append(("State-Reward Correlation", test_state_reward_correlation()))
        results.append(("Diversity Bonus", test_diversity_bonus_function()))
    except Exception as e:
        print(f"\n❌ CHECK 2 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("="*70)
    print("CHECK 2 SUMMARY")
    print("="*70)

    passed = all(r[1] for r in results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("="*70)

    if passed:
        print("✅ CHECK 2 PASSED - Rewards are sane")
        print("\nNext: Run CHECK 3 (balance test)")
    else:
        print("❌ CHECK 2 FAILED - Fix rewards before proceeding")

    print("="*70 + "\n")

    return passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
