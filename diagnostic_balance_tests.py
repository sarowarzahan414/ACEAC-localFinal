"""
Diagnostic Tests for Environment Balance

Tests to identify why Blue wins 100% of the time:
1. Can Red win against passive Blue?
2. Can Blue lose against aggressive Red?
3. What happens with random vs random play?
4. Is the timeout advantage breaking balance?
"""

import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv
from stable_baselines3 import PPO


def test_red_vs_passive_blue(num_battles=20):
    """
    Test 1: Can Red EVER win if Blue does nothing?

    Expected: Red should win 100% if Blue is passive
    If Red loses: Red actions are too weak or Blue starts too strong
    """
    print("\n" + "="*70)
    print("TEST 1: RED VS PASSIVE BLUE")
    print("="*70)
    print("Testing if Red can win when Blue takes no defensive actions...")
    print()

    # Create a dummy passive model that always chooses action 0
    class PassiveModel:
        def predict(self, obs, deterministic=False):
            return 0, None  # Always action 0 (minimal effect)

    passive_blue = PassiveModel()
    red_wins = 0
    blue_wins = 0
    final_states = []

    for battle in range(num_battles):
        env = ZeroSumCyberEnv(agent_role="red", opponent_model=passive_blue)
        obs, _ = env.reset()

        for step in range(100):
            # Red tries to be aggressive - alternate between different attack actions
            action = np.random.randint(0, 25)
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                if info['winner'] == 'red':
                    red_wins += 1
                else:
                    blue_wins += 1
                final_states.append(info['state_mean'])
                break

    print(f"Results ({num_battles} battles):")
    print(f"  Red wins:  {red_wins}/{num_battles} ({red_wins/num_battles*100:.1f}%)")
    print(f"  Blue wins: {blue_wins}/{num_battles} ({blue_wins/num_battles*100:.1f}%)")
    print(f"  Mean final state: {np.mean(final_states):.3f}")
    print()

    if red_wins > num_battles * 0.8:
        print("✓ PASS: Red can win against passive Blue")
        return True
    else:
        print("❌ FAIL: Red cannot win even against passive Blue!")
        print("   This suggests Red actions are too weak or environment is biased.")
        return False


def test_blue_vs_passive_red(num_battles=20):
    """
    Test 2: Can Blue lose if Red does nothing?

    Expected: Blue should win 100% if Red is passive
    Validates Blue can actually win when unopposed
    """
    print("\n" + "="*70)
    print("TEST 2: BLUE VS PASSIVE RED")
    print("="*70)
    print("Testing if Blue wins when Red takes no offensive actions...")
    print()

    class PassiveModel:
        def predict(self, obs, deterministic=False):
            return 0, None

    passive_red = PassiveModel()
    red_wins = 0
    blue_wins = 0
    final_states = []

    for battle in range(num_battles):
        env = ZeroSumCyberEnv(agent_role="blue", opponent_model=passive_red)
        obs, _ = env.reset()

        for step in range(100):
            action = np.random.randint(0, 25)  # Blue defends actively
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                if info['winner'] == 'blue':
                    blue_wins += 1
                else:
                    red_wins += 1
                final_states.append(info['state_mean'])
                break

    print(f"Results ({num_battles} battles):")
    print(f"  Blue wins: {blue_wins}/{num_battles} ({blue_wins/num_battles*100:.1f}%)")
    print(f"  Red wins:  {red_wins}/{num_battles} ({red_wins/num_battles*100:.1f}%)")
    print(f"  Mean final state: {np.mean(final_states):.3f}")
    print()

    if blue_wins > num_battles * 0.8:
        print("✓ PASS: Blue dominates passive Red as expected")
        return True
    else:
        print("⚠️  WARNING: Blue struggles even against passive Red")
        return False


def test_random_vs_random(num_battles=50):
    """
    Test 3: Random vs Random baseline

    Expected: Should be close to 50/50 if environment is balanced
    Actual: Will reveal the timeout advantage
    """
    print("\n" + "="*70)
    print("TEST 3: RANDOM VS RANDOM (BALANCED BASELINE)")
    print("="*70)
    print("Both agents play randomly - should be ~50/50 in balanced game...")
    print()

    class RandomModel:
        def __init__(self, num_actions=25):
            self.num_actions = num_actions

        def predict(self, obs, deterministic=False):
            return np.random.randint(0, self.num_actions), None

    random_blue = RandomModel()
    random_red = RandomModel()

    # Test from Red's perspective
    red_wins = 0
    blue_wins = 0
    draws = 0
    timeouts = 0
    final_states = []
    timeout_states = []

    for battle in range(num_battles):
        env = ZeroSumCyberEnv(agent_role="red", opponent_model=random_blue)
        obs, _ = env.reset()

        for step in range(100):
            action = np.random.randint(0, 25)
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                winner = info['winner']
                final_state = info['state_mean']
                final_states.append(final_state)

                if winner == 'red':
                    red_wins += 1
                elif winner == 'blue':
                    blue_wins += 1
                else:
                    draws += 1

                # Check if this was a timeout
                if step >= 99:  # Last step = timeout
                    timeouts += 1
                    timeout_states.append(final_state)

                break

    print(f"Results ({num_battles} battles):")
    print(f"  Red wins:  {red_wins}/{num_battles} ({red_wins/num_battles*100:.1f}%)")
    print(f"  Blue wins: {blue_wins}/{num_battles} ({blue_wins/num_battles*100:.1f}%)")
    print(f"  Draws:     {draws}/{num_battles} ({draws/num_battles*100:.1f}%)")
    print()
    print(f"State statistics:")
    print(f"  Mean final state: {np.mean(final_states):.3f}")
    print(f"  Std final state:  {np.std(final_states):.3f}")
    print()
    print(f"Timeout analysis:")
    print(f"  Games ending in timeout: {timeouts}/{num_battles} ({timeouts/num_battles*100:.1f}%)")
    if timeout_states:
        print(f"  Mean state at timeout: {np.mean(timeout_states):.3f}")
        print(f"    (0.3-0.7 = contested, <0.3 = Blue wins, >0.7 = Red wins)")
    print()

    if abs(red_wins/num_battles - 0.5) < 0.15:
        print("✓ PASS: Win rate close to 50/50 (balanced)")
        return True
    elif blue_wins > red_wins:
        print("❌ FAIL: Blue wins significantly more than Red with random play!")
        print("   This reveals structural bias toward Blue.")
        print()
        print("   ROOT CAUSE: Timeout condition gives Blue automatic win.")
        print("   When both play equally (random), state stays ~0.5")
        print("   State at 0.5 is not a win for Red (needs >0.7)")
        print("   So Blue wins by timeout!")
        return False
    else:
        print("⚠️  Unexpected: Red wins more (investigate)")
        return False


def test_timeout_advantage(num_battles=30):
    """
    Test 4: Explicitly test the timeout advantage

    Both agents do nothing - should timeout with state at ~0.5
    Reveals Blue's structural advantage
    """
    print("\n" + "="*70)
    print("TEST 4: TIMEOUT ADVANTAGE (THE SMOKING GUN)")
    print("="*70)
    print("Both agents do minimal actions - state should stay ~0.5...")
    print("Testing who wins when state is neutral at timeout...")
    print()

    class MinimalModel:
        def predict(self, obs, deterministic=False):
            return 0, None  # Minimal action

    minimal_blue = MinimalModel()

    red_wins = 0
    blue_wins = 0
    final_states = []

    for battle in range(num_battles):
        env = ZeroSumCyberEnv(agent_role="red", opponent_model=minimal_blue)
        obs, _ = env.reset()

        for step in range(100):
            action = 0  # Red also does minimal action
            obs, reward, terminated, _, info = env.step(action)

            if terminated:
                winner = info['winner']
                final_state = info['state_mean']
                final_states.append(final_state)

                if winner == 'red':
                    red_wins += 1
                elif winner == 'blue':
                    blue_wins += 1

                break

    mean_state = np.mean(final_states)

    print(f"Results ({num_battles} battles):")
    print(f"  Red wins:  {red_wins}/{num_battles}")
    print(f"  Blue wins: {blue_wins}/{num_battles}")
    print(f"  Mean final state: {mean_state:.3f}")
    print()

    if mean_state > 0.4 and mean_state < 0.6:
        print(f"State stayed neutral (~0.5) as expected.")
        print(f"Neutral state is between win thresholds (0.3-0.7)")
        print()

        if blue_wins > num_battles * 0.9:
            print("❌ CONFIRMED: Blue wins 100% when state is neutral!")
            print()
            print("ROOT CAUSE IDENTIFIED:")
            print("  - State at timeout: ~0.5 (neutral/contested)")
            print("  - Red needs >0.7 to win")
            print("  - Blue needs <0.3 to win")
            print("  - Timeout gives Blue automatic win (line 204-205 in environment)")
            print()
            print("This is the structural imbalance causing 100/0 win rate.")
            print("Blue just needs to defend and run out the clock.")
            return False
        else:
            print("✓ Red wins some neutral-state games (unexpected but good)")
            return True
    else:
        print(f"⚠️  State moved significantly even with minimal actions")
        print(f"   (expected ~0.5, got {mean_state:.3f})")
        return False


def run_all_diagnostics():
    """Run all diagnostic tests"""
    print("\n" + "="*70)
    print("ENVIRONMENT BALANCE DIAGNOSTIC SUITE")
    print("="*70)
    print("Testing hypothesis: Blue wins 100% due to timeout advantage")
    print("="*70)

    results = []

    # Test 1: Can Red win at all?
    results.append(("Red vs Passive Blue", test_red_vs_passive_blue(20)))

    # Test 2: Can Blue win? (sanity check)
    results.append(("Blue vs Passive Red", test_blue_vs_passive_red(20)))

    # Test 3: Random baseline
    results.append(("Random vs Random", test_random_vs_random(50)))

    # Test 4: Timeout advantage
    results.append(("Timeout Advantage", test_timeout_advantage(30)))

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()

    if not results[2][1] or not results[3][1]:  # Random vs random failed OR timeout test failed
        print("CONCLUSION:")
        print("-"*70)
        print("Environment is fundamentally UNBALANCED due to timeout advantage.")
        print()
        print("RECOMMENDED FIX:")
        print("  Option 1: Make timeout a DRAW instead of Blue win")
        print("  Option 2: Lower win thresholds (e.g., 0.6/0.4 instead of 0.7/0.3)")
        print("  Option 3: Give Red a starting advantage (initial state ~0.6)")
        print()
        print("Best fix: Option 1 (timeout = draw)")
        print("This removes Blue's structural advantage while keeping win conditions clear.")
    else:
        print("Environment appears balanced. Investigate training dynamics instead.")

    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_diagnostics()
