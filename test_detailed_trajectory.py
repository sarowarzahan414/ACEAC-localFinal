"""
Detailed Step-by-Step Trajectory Analysis

This reveals EXACTLY what happens during an episode:
- How state changes each step
- Whether it's moving toward victory or random walking
- Where the bug actually is
"""

import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv


def detailed_episode_trajectory():
    """Run single episode with detailed logging"""

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)

    print("="*70)
    print("DETAILED SINGLE EPISODE ANALYSIS")
    print("="*70)
    print("Red agent using action 0 consistently (no opponent)")
    print()

    obs, _ = env.reset()
    initial_state = np.mean(env.state)

    print(f"Initial state mean: {initial_state:.6f}")
    print(f"Target for Red win: 0.700000 (decisive)")
    print(f"Target for Blue win: 0.300000 (decisive)")
    print(f"Contested range: 0.300000 - 0.700000 (timeout = draw)")
    print()
    print(f"{'Step':<6} {'State Mean':<12} {'Change':<10} {'Winner':<12} {'Terminated':<12}")
    print("-" * 70)

    state_history = [initial_state]
    change_history = []

    for step in range(100):
        red_action = 0  # Consistent action
        obs, reward, terminated, truncated, info = env.step(red_action)

        current_state = info['state_mean']
        change = current_state - state_history[-1]

        state_history.append(current_state)
        change_history.append(change)

        # Print every step for first 20, then every 10
        if step < 20 or step % 10 == 0 or terminated or truncated:
            print(f"{step:<6} {current_state:.6f}     {change:+.6f}   {info['winner']:<12} {str(terminated):<12}")

        if terminated or truncated:
            print()
            print("="*70)
            print("EPISODE ENDED")
            print("="*70)
            print(f"Final step: {step}")
            print(f"Final state mean: {current_state:.6f}")
            print(f"Winner: {info['winner']}")
            print(f"Total state change: {current_state - initial_state:+.6f}")
            print(f"Average change per step: {np.mean(change_history):+.6f}")
            print(f"Std dev of changes: {np.std(change_history):.6f}")
            break

    # Analysis
    print()
    print("="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)

    # Check if state is consistently moving in one direction
    positive_changes = sum(1 for c in change_history if c > 0)
    negative_changes = sum(1 for c in change_history if c < 0)

    print(f"Positive changes: {positive_changes}/{len(change_history)} ({positive_changes/len(change_history)*100:.1f}%)")
    print(f"Negative changes: {negative_changes}/{len(change_history)} ({negative_changes/len(change_history)*100:.1f}%)")

    # Check for random walk vs directed movement
    if abs(positive_changes - negative_changes) < len(change_history) * 0.2:
        print("⚠️  DIAGNOSIS: Random walk detected (50/50 up/down)")
        print("   Noise is overwhelming action effects")
    elif positive_changes > negative_changes * 1.5:
        print("✓ DIAGNOSIS: Directed upward movement")
        print("   Actions are working, but may be too weak or slow")
    else:
        print("? DIAGNOSIS: Unclear pattern")

    # Check expected vs actual movement
    expected_change_per_step = 0.025  # Theory: 8 dims * 0.10 / 32 dims
    actual_change_per_step = np.mean(change_history)

    print()
    print(f"Expected change per step (theory): +{expected_change_per_step:.6f}")
    print(f"Actual change per step (measured): {actual_change_per_step:+.6f}")
    print(f"Ratio (actual/expected): {actual_change_per_step/expected_change_per_step:.2f}")

    if abs(actual_change_per_step) < expected_change_per_step * 0.3:
        print("❌ PROBLEM: Actual movement is <30% of expected")
        print("   Likely causes:")
        print("   1. Noise is too high (drowning out signal)")
        print("   2. Action effects are incorrectly implemented")
        print("   3. State decay/saturation hidden somewhere")

    # Estimate steps needed to reach 0.7
    distance_to_win = 0.7 - initial_state
    if actual_change_per_step > 0.0001:
        steps_needed = distance_to_win / actual_change_per_step
        print()
        print(f"Distance to Red win threshold: {distance_to_win:.6f}")
        print(f"Steps needed at current rate: {steps_needed:.1f}")
        if steps_needed > 100:
            print(f"❌ PROBLEM: Need {steps_needed:.0f} steps but max is 100")
            print("   Red can NEVER win at this rate!")

    return state_history, change_history


def plot_trajectory(state_history):
    """Plot state trajectory if matplotlib available"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # Main trajectory
        plt.subplot(1, 2, 1)
        plt.plot(state_history, marker='o', markersize=3, linewidth=1)
        plt.axhline(y=0.7, color='r', linestyle='--', linewidth=2, label='Red win (0.7)')
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1, label='Neutral (0.5)')
        plt.axhline(y=0.3, color='b', linestyle='--', linewidth=2, label='Blue win (0.3)')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('State Mean', fontsize=12)
        plt.title('State Trajectory Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.2, 0.8)

        # Change per step
        plt.subplot(1, 2, 2)
        changes = [state_history[i+1] - state_history[i] for i in range(len(state_history)-1)]
        plt.plot(changes, marker='o', markersize=2, linewidth=0.5, alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axhline(y=0.025, color='g', linestyle='--', alpha=0.5, label='Expected (+0.025)')
        plt.axhline(y=-0.025, color='r', linestyle='--', alpha=0.5, label='Expected (-0.025)')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Change in State Mean', fontsize=12)
        plt.title('State Change Per Step', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('detailed_trajectory.png', dpi=150, bbox_inches='tight')
        print()
        print("✓ Plot saved to: detailed_trajectory.png")

    except ImportError:
        print()
        print("(matplotlib not available - skipping plot)")


def run_multiple_episodes(num_episodes=10):
    """Run multiple episodes to check consistency"""

    print("\n" + "="*70)
    print(f"RUNNING {num_episodes} EPISODES TO CHECK CONSISTENCY")
    print("="*70)

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)

    final_states = []
    total_changes = []
    winners = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        initial = np.mean(env.state)

        for step in range(100):
            obs, reward, done, truncated, info = env.step(0)  # Always action 0
            if done or truncated:
                final = info['state_mean']
                final_states.append(final)
                total_changes.append(final - initial)
                winners.append(info['winner'])
                break

    # Summary
    print(f"\nResults from {num_episodes} episodes:")
    print(f"  Red wins: {winners.count('red')} ({winners.count('red')/num_episodes*100:.1f}%)")
    print(f"  Blue wins: {winners.count('blue')} ({winners.count('blue')/num_episodes*100:.1f}%)")
    print(f"  Draws: {winners.count('draw')} ({winners.count('draw')/num_episodes*100:.1f}%)")
    print()
    print(f"Final state statistics:")
    print(f"  Mean: {np.mean(final_states):.6f}")
    print(f"  Std:  {np.std(final_states):.6f}")
    print(f"  Min:  {np.min(final_states):.6f}")
    print(f"  Max:  {np.max(final_states):.6f}")
    print()
    print(f"Total change statistics:")
    print(f"  Mean: {np.mean(total_changes):+.6f}")
    print(f"  Std:  {np.std(total_changes):.6f}")

    # Diagnosis
    if winners.count('draw') > num_episodes * 0.8:
        print()
        print("❌ PROBLEM: >80% draws")
        print("   Environment allows victories but they're not happening")
        print("   Likely cause: Action effects too weak or noise too high")

    if np.max(final_states) < 0.6:
        print()
        print("❌ PROBLEM: Never exceeded 0.6 state")
        print("   Red can't reach 0.7 threshold at current movement rate")

    if abs(np.mean(total_changes)) < 0.05:
        print()
        print("❌ PROBLEM: Average total change < 0.05")
        print("   State barely moves from initial position")
        print("   This confirms random walk hypothesis")


if __name__ == "__main__":
    print("\n" + "!"*70)
    print("DETAILED TRAJECTORY DIAGNOSTIC")
    print("!"*70)
    print()
    print("This test reveals exactly what happens during episodes.")
    print("It will show whether the environment bug is:")
    print("  A) Random walk (noise overwhelming signal)")
    print("  B) Too-weak actions (correct direction but too slow)")
    print("  C) Hidden decay/saturation")
    print("  D) Something else entirely")
    print()

    # Run detailed single episode
    state_history, change_history = detailed_episode_trajectory()

    # Try to plot
    plot_trajectory(state_history)

    # Run multiple episodes for consistency
    run_multiple_episodes(num_episodes=20)

    print()
    print("="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
