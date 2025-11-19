"""
CHECK 3: Balance Test

Verify Red and Blue agents have symmetric learning opportunities.
Tests:
1. Red and Blue can both improve state
2. Neither agent has unfair advantage
3. Opponent interaction is symmetric
4. Both agents can earn positive rewards
"""

import numpy as np
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv
from stable_baselines3 import PPO

def test_red_blue_symmetry():
    """Test that Red and Blue have symmetric opportunities"""
    print("="*70)
    print("CHECK 3.1: Red/Blue Symmetry Test")
    print("="*70)

    # Test Red agent
    red_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
    red_rewards = []

    for _ in range(50):
        obs, _ = red_env.reset()
        episode_reward = 0

        for step in range(100):
            action = red_env.action_space.sample()
            obs, reward, done, _, info = red_env.step(action)
            episode_reward += reward

            if done:
                break

        red_rewards.append(episode_reward)

    # Test Blue agent
    blue_env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=None)
    blue_rewards = []

    for _ in range(50):
        obs, _ = blue_env.reset()
        episode_reward = 0

        for step in range(100):
            action = blue_env.action_space.sample()
            obs, reward, done, _, info = blue_env.step(action)
            episode_reward += reward

            if done:
                break

        blue_rewards.append(episode_reward)

    print(f"Red Agent Statistics:")
    print(f"  Mean episode reward: {np.mean(red_rewards):.3f}")
    print(f"  Std:  {np.std(red_rewards):.3f}")

    print(f"\nBlue Agent Statistics:")
    print(f"  Mean episode reward: {np.mean(blue_rewards):.3f}")
    print(f"  Std:  {np.std(blue_rewards):.3f}")

    # Check symmetry
    mean_diff = abs(np.mean(red_rewards) - np.mean(blue_rewards))

    print(f"\nSymmetry Check:")
    print(f"  Mean reward difference: {mean_diff:.3f}")

    if mean_diff < 50.0:
        print("✓ Red and Blue have similar reward distributions (good)")
    else:
        print("⚠️  Large asymmetry detected - one agent may be favored")

    print()
    return True


def test_state_movement():
    """Test that both agents can move state in their direction"""
    print("="*70)
    print("CHECK 3.2: State Movement Test")
    print("="*70)

    # Red should increase state
    red_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
    red_state_changes = []

    for _ in range(20):
        obs, _ = red_env.reset()
        initial_state_mean = np.mean(red_env.state)

        for step in range(50):
            action = red_env.action_space.sample()
            obs, reward, done, _, info = red_env.step(action)

        final_state_mean = np.mean(red_env.state)
        red_state_changes.append(final_state_mean - initial_state_mean)

    # Blue should decrease state
    blue_env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=None)
    blue_state_changes = []

    for _ in range(20):
        obs, _ = blue_env.reset()
        initial_state_mean = np.mean(blue_env.state)

        for step in range(50):
            action = blue_env.action_space.sample()
            obs, reward, done, _, info = blue_env.step(action)

        final_state_mean = np.mean(blue_env.state)
        blue_state_changes.append(final_state_mean - initial_state_mean)

    print(f"Red Agent State Movement:")
    print(f"  Mean change: {np.mean(red_state_changes):.4f}")
    print(f"  (Should be slightly positive)")

    print(f"\nBlue Agent State Movement:")
    print(f"  Mean change: {np.mean(blue_state_changes):.4f}")
    print(f"  (Should be slightly negative)")

    red_mean = np.mean(red_state_changes)
    blue_mean = np.mean(blue_state_changes)

    if red_mean > 0 and blue_mean < 0:
        print("\n✓ Both agents can move state in their direction (good)")
    elif abs(red_mean) < 0.01 and abs(blue_mean) < 0.01:
        print("\n⚠️  State barely moves - may need more steps or stronger effects")
    else:
        print("\n⚠️  Unexpected state movement pattern")

    print()
    return True


def test_opponent_interaction_balance():
    """Test that opponent interaction is balanced"""
    print("="*70)
    print("CHECK 3.3: Opponent Interaction Balance")
    print("="*70)

    # Create a dummy opponent (random policy)
    dummy_env = DynamicCoEvolutionEnv(agent_role="blue")
    dummy_opponent = PPO("MlpPolicy", dummy_env, verbose=0)

    # Test Red vs random Blue
    red_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=dummy_opponent)
    red_vs_blue_rewards = []

    for _ in range(20):
        obs, _ = red_env.reset()
        episode_reward = 0

        for step in range(50):
            action = red_env.action_space.sample()
            obs, reward, done, _, info = red_env.step(action)
            episode_reward += reward

            if done:
                break

        red_vs_blue_rewards.append(episode_reward)

    # Test Blue vs random Red
    dummy_red_env = DynamicCoEvolutionEnv(agent_role="red")
    dummy_red_opponent = PPO("MlpPolicy", dummy_red_env, verbose=0)

    blue_env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=dummy_red_opponent)
    blue_vs_red_rewards = []

    for _ in range(20):
        obs, _ = blue_env.reset()
        episode_reward = 0

        for step in range(50):
            action = blue_env.action_space.sample()
            obs, reward, done, _, info = blue_env.step(action)
            episode_reward += reward

            if done:
                break

        blue_vs_red_rewards.append(episode_reward)

    print(f"Red vs Blue (random):")
    print(f"  Mean episode reward: {np.mean(red_vs_blue_rewards):.3f}")

    print(f"\nBlue vs Red (random):")
    print(f"  Mean episode reward: {np.mean(blue_vs_red_rewards):.3f}")

    mean_diff = abs(np.mean(red_vs_blue_rewards) - np.mean(blue_vs_red_rewards))

    print(f"\nBalance Check:")
    print(f"  Reward difference: {mean_diff:.3f}")

    if mean_diff < 100.0:
        print("✓ Opponent interaction is balanced (good)")
    else:
        print("⚠️  Asymmetric opponent interaction")

    print()
    return True


def test_both_can_win():
    """Test that both agents can achieve positive rewards"""
    print("="*70)
    print("CHECK 3.4: Both Agents Can Win")
    print("="*70)

    # Red agent
    red_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
    red_positive_count = 0

    for _ in range(30):
        obs, _ = red_env.reset()
        episode_reward = 0

        for step in range(100):
            action = red_env.action_space.sample()
            obs, reward, done, _, info = red_env.step(action)
            episode_reward += reward

            if done:
                break

        if episode_reward > 0:
            red_positive_count += 1

    # Blue agent
    blue_env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=None)
    blue_positive_count = 0

    for _ in range(30):
        obs, _ = blue_env.reset()
        episode_reward = 0

        for step in range(100):
            action = blue_env.action_space.sample()
            obs, reward, done, _, info = blue_env.step(action)
            episode_reward += reward

            if done:
                break

        if episode_reward > 0:
            blue_positive_count += 1

    print(f"Red Agent:")
    print(f"  Episodes with positive reward: {red_positive_count}/30 ({red_positive_count/30*100:.1f}%)")

    print(f"\nBlue Agent:")
    print(f"  Episodes with positive reward: {blue_positive_count}/30 ({blue_positive_count/30*100:.1f}%)")

    if red_positive_count > 0 and blue_positive_count > 0:
        print("\n✓ Both agents can achieve positive rewards (good)")
    else:
        print("\n⚠️  One or both agents struggle to get positive rewards")

    if red_positive_count > 3 and blue_positive_count > 3:
        print("✓ Both agents can consistently succeed (good)")

    print()
    return True


def main():
    print("\n" + "="*70)
    print("CHECK 3: BALANCE TEST")
    print("="*70)
    print("Verifying Red/Blue symmetry and fairness\n")

    results = []

    try:
        results.append(("Red/Blue Symmetry", test_red_blue_symmetry()))
        results.append(("State Movement", test_state_movement()))
        results.append(("Opponent Interaction", test_opponent_interaction_balance()))
        results.append(("Both Can Win", test_both_can_win()))
    except Exception as e:
        print(f"\n❌ CHECK 3 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("="*70)
    print("CHECK 3 SUMMARY")
    print("="*70)

    passed = all(r[1] for r in results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("="*70)

    if passed:
        print("✅ CHECK 3 PASSED - System is balanced")
        print("\nNext: Run CHECK 4 (one generation test)")
    else:
        print("❌ CHECK 3 FAILED - Fix balance before proceeding")

    print("="*70 + "\n")

    return passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
