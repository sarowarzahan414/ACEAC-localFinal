"""
Quick test to verify timeout = draw fix

This directly tests the _check_terminal_conditions method
to prove the fix works without running full training.
"""

import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv


def test_timeout_is_draw():
    """Verify that timeout produces draw when state is contested"""
    print("\n" + "="*70)
    print("QUICK TIMEOUT VALIDATION TEST")
    print("="*70)
    print("Testing that timeout produces DRAW for contested states...")
    print()

    env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)

    # Test various contested state values at timeout
    test_cases = [
        (0.30, "Should be draw (just above Blue threshold)"),
        (0.35, "Should be draw (clearly contested)"),
        (0.40, "Should be draw (contested)"),
        (0.45, "Should be draw (contested)"),
        (0.50, "Should be draw (neutral)"),
        (0.55, "Should be draw (contested)"),
        (0.60, "Should be draw (contested)"),
        (0.65, "Should be draw (contested)"),
        (0.70, "Should be draw (just below Red threshold)"),
    ]

    print("Testing timeout outcomes at various state values:")
    print("-" * 70)

    all_draws = True

    for state_value, description in test_cases:
        env.reset()
        env.state = np.full(env.state_dim, state_value, dtype=np.float32)
        env.step_count = env.max_steps  # Force timeout

        terminated, winner = env._check_terminal_conditions()

        status = "✓" if winner == "draw" else "❌"
        result = f"{status} State: {state_value:.2f} -> Winner: {winner:6s} | {description}"
        print(result)

        if winner != "draw":
            all_draws = False
            print(f"   ERROR: Expected draw but got {winner}!")

    print()

    # Test decisive victories still work
    print("Testing decisive victories still work:")
    print("-" * 70)

    decisive_tests = [
        (0.25, "blue", "Blue decisive (< 0.3)"),
        (0.29, "blue", "Blue decisive (< 0.3)"),
        (0.71, "red", "Red decisive (> 0.7)"),
        (0.85, "red", "Red decisive (> 0.7)"),
    ]

    decisive_ok = True

    for state_value, expected_winner, description in decisive_tests:
        env.reset()
        env.state = np.full(env.state_dim, state_value, dtype=np.float32)
        env.step_count = 50  # Not at timeout

        terminated, winner = env._check_terminal_conditions()

        status = "✓" if winner == expected_winner else "❌"
        result = f"{status} State: {state_value:.2f} -> Winner: {winner:6s} | {description}"
        print(result)

        if winner != expected_winner:
            decisive_ok = False
            print(f"   ERROR: Expected {expected_winner} but got {winner}!")

    print()
    print("="*70)

    if all_draws and decisive_ok:
        print("✅ SUCCESS: Timeout = draw fix is working correctly!")
        print()
        print("Summary:")
        print("  ✓ All contested states (0.3-0.7) produce draws at timeout")
        print("  ✓ Decisive victories (< 0.3, > 0.7) still work")
        print("  ✓ No arbitrary thresholds")
        print("  ✓ No Blue timeout advantage")
        print()
        print("Ready to proceed with training!")
        return True
    else:
        print("❌ FAILED: Fix not working as expected")
        print()
        if not all_draws:
            print("  Problem: Some contested states not producing draws")
        if not decisive_ok:
            print("  Problem: Decisive victories not working")
        return False

    print("="*70)


if __name__ == "__main__":
    import sys
    success = test_timeout_is_draw()
    sys.exit(0 if success else 1)
