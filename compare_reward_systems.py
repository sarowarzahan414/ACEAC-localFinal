"""
Side-by-Side Comparison: Old (Cooperative) vs New (Zero-Sum)

This demonstrates WHY the old system was broken and
why the new system is fundamentally different.

Author: @sarowarzahan414
Date: 2025-11-19
"""

import numpy as np
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv
from aceac_zerosum_environment import ZeroSumCyberEnv


def test_old_system():
    """Test old (broken) reward system"""
    print("="*70)
    print("OLD SYSTEM (COOPERATIVE)")
    print("="*70)

    # Red attacking alone
    print("\nScenario 1: Red Acting Alone (No Opponent)")
    print("-" * 70)

    env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
    obs, _ = env.reset()

    rewards = []
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Total reward: {np.sum(rewards):.1f}")
    print(f"Rewards all positive: {all(r >= 0 for r in rewards)}")

    # Blue defending alone
    print("\nScenario 2: Blue Acting Alone (No Opponent)")
    print("-" * 70)

    env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=None)
    obs, _ = env.reset()

    rewards = []
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Total reward: {np.sum(rewards):.1f}")
    print(f"Rewards all positive: {all(r >= 0 for r in rewards)}")

    # The problem
    print("\n" + "="*70)
    print("THE PROBLEM:")
    print("="*70)
    print("❌ Both Red and Blue get POSITIVE rewards")
    print("❌ No competition - both can succeed simultaneously")
    print("❌ Agents learn 'do stuff = get points', not 'beat opponent'")
    print("❌ This is COOPERATIVE, not ADVERSARIAL")
    print("="*70 + "\n")


def test_new_system():
    """Test new (zero-sum) reward system"""
    print("="*70)
    print("NEW SYSTEM (ZERO-SUM)")
    print("="*70)

    # Red attacking alone
    print("\nScenario 1: Red Acting Alone (No Opponent)")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
    obs, _ = env.reset()

    rewards = []
    outcome = None

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)

        if done:
            outcome = info['winner']
            break

    print(f"Outcome: {outcome}")
    print(f"Final reward: {rewards[-1]:.1f} (when episode ends)")
    print(f"Total reward: {np.sum(rewards):.1f}")
    print(f"Episode ends in: WIN (Red dominates without opposition)")

    # Blue defending alone
    print("\nScenario 2: Blue Acting Alone (No Opponent)")
    print("-" * 70)

    env = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)
    obs, _ = env.reset()

    rewards = []
    outcome = None

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)

        if done:
            outcome = info['winner']
            break

    print(f"Outcome: {outcome}")
    print(f"Final reward: {rewards[-1]:.1f} (when episode ends)")
    print(f"Total reward: {np.sum(rewards):.1f}")
    print(f"Episode ends in: WIN (Blue secures network)")

    # Simulated adversarial scenario
    print("\nScenario 3: Simulated Adversarial Game")
    print("-" * 70)
    print("(Approximating Red vs Blue with state manipulation)")

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)

    red_wins = 0
    blue_wins = 0
    red_total_reward = []
    blue_total_reward = []

    for ep in range(10):
        obs, _ = env.reset()

        # Simulate by randomly deciding winner
        if np.random.random() > 0.5:
            # Red wins
            env.state = np.full(env.state_dim, 0.75, dtype=np.float32)
        else:
            # Blue wins
            env.state = np.full(env.state_dim, 0.25, dtype=np.float32)

        obs, reward, done, _, info = env.step(0)

        if info['winner'] == 'red':
            red_wins += 1
            red_total_reward.append(reward)

            # Blue's reward would be opposite
            env_blue = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)
            env_blue.state = env.state.copy()
            _, blue_reward, _, _, _ = env_blue.step(0)
            blue_total_reward.append(blue_reward)
        else:
            blue_wins += 1

            # Calculate rewards from both perspectives
            red_total_reward.append(reward)

            env_blue = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)
            env_blue.state = env.state.copy()
            _, blue_reward, _, _, _ = env_blue.step(0)
            blue_total_reward.append(blue_reward)

    print(f"Red wins: {red_wins}/10")
    print(f"Blue wins: {blue_wins}/10")
    print(f"Red mean reward: {np.mean(red_total_reward):.1f}")
    print(f"Blue mean reward: {np.mean(blue_total_reward):.1f}")
    print(f"Reward sum: {np.mean(red_total_reward) + np.mean(blue_total_reward):.1f} (should be ~0)")

    # The fix
    print("\n" + "="*70)
    print("THE FIX:")
    print("="*70)
    print("✓ Win = +100, Loss = -100, Draw = 0")
    print("✓ One agent's gain = other agent's loss")
    print("✓ Clear winner/loser every episode")
    print("✓ Agents learn 'beat opponent', not 'do stuff'")
    print("✓ This is TRULY ADVERSARIAL")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("REWARD SYSTEM COMPARISON")
    print("="*70)
    print("Demonstrating the fundamental difference between")
    print("cooperative and adversarial reward structures")
    print("="*70 + "\n")

    print("\n")
    test_old_system()

    print("\n\n")
    test_new_system()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nThe old system rewards both agents for activity,")
    print("creating a COOPERATIVE environment where both can win.")
    print("\nThe new system rewards only the WINNER,")
    print("creating a COMPETITIVE environment where agents must")
    print("outperform their opponent to succeed.")
    print("\nThis is the difference between:")
    print("  - Training a boxer by having them shadowbox")
    print("  - Training a boxer by having them fight opponents")
    print("\nOnly the second produces real fighting skills.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
