"""
Critical Validation Tests for ACEAC

This script answers the fundamental question:
"Do trained agents actually learn anything useful?"

Tests:
1. Trained vs Random - Should win >70%
2. Trained vs Greedy - Should win >60%
3. Trained vs Passive - Should win >95%
4. Multi-seed validation - Results should be consistent

If any test fails, training doesn't work.
"""

import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv
from baseline_agents import RandomAgent, GreedyAgent, PassiveAgent
import json
from typing import Dict, List
import os


def evaluate_against_baseline(
    trained_model,
    trained_role: str,
    baseline_agent,
    baseline_role: str,
    num_episodes: int = 100,
    verbose: bool = False
) -> Dict:
    """
    Evaluate trained agent against baseline

    Returns:
        Dict with win rates, rewards, and statistics
    """
    if trained_role == baseline_role:
        raise ValueError("Trained and baseline must be different roles")

    trained_wins = 0
    baseline_wins = 0
    draws = 0

    trained_rewards = []
    baseline_rewards = []

    episode_lengths = []

    for ep in range(num_episodes):
        # Create environment from trained agent's perspective
        if trained_role == "red":
            env = ZeroSumCyberEnv(agent_role="red", opponent_model=baseline_agent)
            trained_is_red = True
        else:
            env = ZeroSumCyberEnv(agent_role="blue", opponent_model=baseline_agent)
            trained_is_red = False

        obs, _ = env.reset()
        trained_total_reward = 0

        for step in range(100):
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            trained_total_reward += reward

            if terminated:
                episode_lengths.append(step + 1)
                winner = info['winner']

                if winner == trained_role:
                    trained_wins += 1
                elif winner == baseline_role:
                    baseline_wins += 1
                else:
                    draws += 1

                trained_rewards.append(trained_total_reward)
                baseline_rewards.append(-trained_total_reward)  # Zero-sum

                break

        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"Trained {trained_wins}, Baseline {baseline_wins}, Draws {draws}")

    results = {
        'trained_agent': trained_role,
        'baseline_agent': baseline_agent.name,
        'num_episodes': num_episodes,
        'trained_wins': trained_wins,
        'baseline_wins': baseline_wins,
        'draws': draws,
        'trained_win_rate': trained_wins / num_episodes,
        'baseline_win_rate': baseline_wins / num_episodes,
        'draw_rate': draws / num_episodes,
        'trained_mean_reward': float(np.mean(trained_rewards)),
        'baseline_mean_reward': float(np.mean(baseline_rewards)),
        'trained_reward_std': float(np.std(trained_rewards)),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'episode_length_std': float(np.std(episode_lengths))
    }

    return results


def print_validation_results(results: Dict, test_name: str):
    """Print formatted results"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"\nOutcomes:")
    print(f"  Trained ({results['trained_agent']}) wins: {results['trained_wins']:3d} "
          f"({results['trained_win_rate']*100:5.1f}%)")
    print(f"  Baseline ({results['baseline_agent']:20s}) wins: {results['baseline_wins']:3d} "
          f"({results['baseline_win_rate']*100:5.1f}%)")
    print(f"  Draws:                       {results['draws']:3d} "
          f"({results['draw_rate']*100:5.1f}%)")

    print(f"\nRewards:")
    print(f"  Trained mean: {results['trained_mean_reward']:+7.1f} "
          f"(std: {results['trained_reward_std']:5.1f})")
    print(f"  Baseline mean: {results['baseline_mean_reward']:+7.1f}")

    print(f"\nEpisode length: {results['avg_episode_length']:.1f} ± "
          f"{results['episode_length_std']:.1f} steps")


def assess_validation_result(results: Dict, test_name: str) -> bool:
    """
    Determine if results pass validation

    Returns:
        True if pass, False if fail
    """
    baseline_type = results['baseline_agent'].lower()

    # Define success criteria for each baseline
    if 'random' in baseline_type:
        # Should beat random >70%
        threshold = 0.70
        required = "Beat Random >70%"
    elif 'greedy' in baseline_type:
        # Should beat greedy >60%
        threshold = 0.60
        required = "Beat Greedy >60%"
    elif 'passive' in baseline_type:
        # Should beat passive >95%
        threshold = 0.95
        required = "Beat Passive >95%"
    else:
        # Unknown baseline
        threshold = 0.55
        required = "Beat baseline >55%"

    win_rate = results['trained_win_rate']
    passed = win_rate > threshold

    print(f"\n{'─'*70}")
    print(f"ASSESSMENT: {test_name}")
    print(f"{'─'*70}")
    print(f"Requirement: {required}")
    print(f"Achieved: {win_rate*100:.1f}% win rate")

    if passed:
        print(f"✅ PASS - Training shows meaningful learning")
    else:
        print(f"❌ FAIL - Training did not produce useful strategies")
        print(f"   Gap: Need {threshold*100:.1f}%, got {win_rate*100:.1f}% "
              f"({(threshold - win_rate)*100:.1f}% short)")

    return passed


def run_all_validations(red_model, blue_model, num_episodes: int = 100):
    """
    Run complete validation suite

    Tests both Red and Blue agents against all baselines
    """
    print("\n" + "="*70)
    print("ACEAC TRAINING VALIDATION SUITE")
    print("="*70)
    print("Testing if trained agents learned meaningful strategies")
    print("="*70)

    all_results = []
    all_passed = True

    # Test Red agent
    print("\n" + "▶"*35)
    print("TESTING RED AGENT (Attacker)")
    print("▶"*35)

    # Red vs Random Blue
    test1 = evaluate_against_baseline(
        red_model, "red",
        RandomAgent(), "blue",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test1, "Red Agent vs Random Blue")
    passed1 = assess_validation_result(test1, "Red vs Random")
    all_results.append(('Red vs Random', test1, passed1))
    all_passed = all_passed and passed1

    # Red vs Greedy Blue
    test2 = evaluate_against_baseline(
        red_model, "red",
        GreedyAgent(role="blue"), "blue",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test2, "Red Agent vs Greedy Blue")
    passed2 = assess_validation_result(test2, "Red vs Greedy")
    all_results.append(('Red vs Greedy', test2, passed2))
    all_passed = all_passed and passed2

    # Red vs Passive Blue
    test3 = evaluate_against_baseline(
        red_model, "red",
        PassiveAgent(), "blue",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test3, "Red Agent vs Passive Blue")
    passed3 = assess_validation_result(test3, "Red vs Passive")
    all_results.append(('Red vs Passive', test3, passed3))
    all_passed = all_passed and passed3

    # Test Blue agent
    print("\n" + "▶"*35)
    print("TESTING BLUE AGENT (Defender)")
    print("▶"*35)

    # Blue vs Random Red
    test4 = evaluate_against_baseline(
        blue_model, "blue",
        RandomAgent(), "red",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test4, "Blue Agent vs Random Red")
    passed4 = assess_validation_result(test4, "Blue vs Random")
    all_results.append(('Blue vs Random', test4, passed4))
    all_passed = all_passed and passed4

    # Blue vs Greedy Red
    test5 = evaluate_against_baseline(
        blue_model, "blue",
        GreedyAgent(role="red"), "red",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test5, "Blue Agent vs Greedy Red")
    passed5 = assess_validation_result(test5, "Blue vs Greedy")
    all_results.append(('Blue vs Greedy', test5, passed5))
    all_passed = all_passed and passed5

    # Blue vs Passive Red
    test6 = evaluate_against_baseline(
        blue_model, "blue",
        PassiveAgent(), "red",
        num_episodes=num_episodes,
        verbose=True
    )
    print_validation_results(test6, "Blue Agent vs Passive Red")
    passed6 = assess_validation_result(test6, "Blue vs Passive")
    all_results.append(('Blue vs Passive', test6, passed6))
    all_passed = all_passed and passed6

    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for test_name, results, passed in all_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        wr = results['trained_win_rate']
        print(f"{status} {test_name:25s} - {wr*100:5.1f}% win rate")

    print()
    if all_passed:
        print("="*70)
        print("🎉 ALL TESTS PASSED - Training produces meaningful learning!")
        print("="*70)
        print("\nYour agents demonstrate:")
        print("  ✓ Better than random play")
        print("  ✓ Better than simple heuristics")
        print("  ✓ Better than passive strategies")
        print("\nThis validates that co-evolutionary training works.")
    else:
        print("="*70)
        print("⚠️  SOME TESTS FAILED - Training quality questionable")
        print("="*70)
        print("\nRecommendations:")
        print("  - Increase training time (more generations)")
        print("  - Adjust hyperparameters (learning rate, entropy)")
        print("  - Check reward function balance")
        print("  - Verify environment implementation")

    # Save results
    os.makedirs("validation_results", exist_ok=True)
    with open("validation_results/baseline_comparison.json", "w") as f:
        json.dump([r[1] for r in all_results], f, indent=2)

    print(f"\nResults saved to: validation_results/baseline_comparison.json")

    return all_passed, all_results


if __name__ == "__main__":
    print("\n" + "!"*70)
    print("ERROR: No trained models provided")
    print("!"*70)
    print("\nThis script requires trained models to validate.")
    print("\nUsage:")
    print("  from validate_training import run_all_validations")
    print("  from stable_baselines3 import PPO")
    print()
    print("  red_model = PPO.load('models/aceac_red_zerosum.zip')")
    print("  blue_model = PPO.load('models/aceac_blue_zerosum.zip')")
    print()
    print("  passed, results = run_all_validations(red_model, blue_model)")
    print()
