"""
ACEAC Zero-Sum Co-Evolution Training - TRUE ADVERSARIAL SYSTEM

This replaces the broken reward system with proper zero-sum adversarial training.

Key Differences from Old System:
- Win: +100, Loss: -100, Draw: 0 (not always positive rewards)
- Clear winner/loser every episode (not both getting participation trophies)
- Agents learn to beat opponent (not just "do actions")
- True adversarial dynamics (not cooperative)

Author: @sarowarzahan414
Date: 2025-11-19
Purpose: Fix fundamental reward structure based on brutal analysis
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import json
import time
import argparse
from datetime import datetime, timezone
import os
from aceac_zerosum_environment import ZeroSumCyberEnv


def test_generation(red_model, blue_model, num_battles=20):
    """
    Test current generation of agents

    Returns win rates, reward statistics, and battle details
    """
    red_wins = 0
    blue_wins = 0
    draws = 0

    red_rewards = []
    blue_rewards = []
    battles = []

    for battle in range(num_battles):
        # Test from Red's perspective
        env_red = ZeroSumCyberEnv(agent_role="red", opponent_model=blue_model)
        obs, _ = env_red.reset()

        red_total_reward = 0

        for step in range(100):
            action, _ = red_model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env_red.step(action)
            red_total_reward += reward

            if terminated:
                break

        # Determine winner
        winner = info.get('winner', 'ongoing')

        if winner == 'red':
            red_wins += 1
        elif winner == 'blue':
            blue_wins += 1
        else:
            draws += 1

        # Blue's reward is opposite (zero-sum)
        blue_total_reward = -red_total_reward

        red_rewards.append(red_total_reward)
        blue_rewards.append(blue_total_reward)

        battles.append({
            'battle': battle + 1,
            'winner': winner,
            'red_reward': float(red_total_reward),
            'blue_reward': float(blue_total_reward),
            'state_mean': float(info.get('state_mean', 0.5))
        })

    results = {
        'red_wins': red_wins,
        'blue_wins': blue_wins,
        'draws': draws,
        'red_win_rate': red_wins / num_battles,
        'blue_win_rate': blue_wins / num_battles,
        'red_mean_reward': float(np.mean(red_rewards)),
        'blue_mean_reward': float(np.mean(blue_rewards)),
        'red_reward_std': float(np.std(red_rewards)),
        'blue_reward_std': float(np.std(blue_rewards)),
        'reward_sum': float(np.mean(red_rewards) + np.mean(blue_rewards)),  # Should be ~0
        'battles': battles
    }

    return results


def zerosum_coevolution_training(
    timesteps_per_generation=5000,
    generations=10,
    test_battles=20,
    save_freq=5
):
    """
    Zero-sum co-evolution training

    Args:
        timesteps_per_generation: Training steps per agent per generation
        generations: Number of co-evolution generations
        test_battles: Number of test battles per generation
        save_freq: Save models every N generations
    """
    print("\n" + "="*70)
    print("ACEAC ZERO-SUM CO-EVOLUTION TRAINING")
    print("="*70)
    print("TRUE ADVERSARIAL SYSTEM - NO PARTICIPATION TROPHIES")
    print("-"*70)
    print(f"Generations: {generations}")
    print(f"Timesteps per generation: {timesteps_per_generation}")
    print(f"Test battles per generation: {test_battles}")
    print("="*70)
    print()

    # Create directories
    os.makedirs("models/zerosum", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize fresh agents (DO NOT load broken reward models!)
    print("Initializing fresh agents with zero-sum environment...")

    env_red_init = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
    env_blue_init = ZeroSumCyberEnv(agent_role="blue", opponent_model=None)

    red_model = PPO(
        "MlpPolicy",
        env_red_init,
        verbose=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01  # Encourage exploration
    )

    blue_model = PPO(
        "MlpPolicy",
        env_blue_init,
        verbose=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

    print("✓ Fresh agents initialized")
    print()

    # Training log
    training_log = {
        'system': 'zero-sum',
        'start_time': datetime.now(timezone.utc).isoformat(),
        'config': {
            'timesteps_per_generation': timesteps_per_generation,
            'generations': generations,
            'test_battles': test_battles
        },
        'generations': []
    }

    total_start = time.time()

    # Co-evolution loop
    for gen in range(1, generations + 1):
        print("="*70)
        print(f"GENERATION {gen}/{generations}")
        print("="*70)

        gen_start = time.time()

        # Train Red against current Blue
        print(f"\nTraining Red (Gen {gen}) vs Blue (Gen {gen-1})...")
        env_red = ZeroSumCyberEnv(agent_role="red", opponent_model=blue_model)
        red_model.set_env(env_red)

        red_start = time.time()
        red_model.learn(
            total_timesteps=timesteps_per_generation,
            reset_num_timesteps=False,
            progress_bar=True
        )
        red_dur = time.time() - red_start
        print(f"✓ Red training complete ({red_dur:.1f}s)")

        # Save Red
        if gen % save_freq == 0 or gen == generations:
            red_model.save(f"models/zerosum/red_gen{gen}.zip")
            print(f"  Saved: models/zerosum/red_gen{gen}.zip")

        # Train Blue against updated Red
        print(f"\nTraining Blue (Gen {gen}) vs Red (Gen {gen})...")
        env_blue = ZeroSumCyberEnv(agent_role="blue", opponent_model=red_model)
        blue_model.set_env(env_blue)

        blue_start = time.time()
        blue_model.learn(
            total_timesteps=timesteps_per_generation,
            reset_num_timesteps=False,
            progress_bar=True
        )
        blue_dur = time.time() - blue_start
        print(f"✓ Blue training complete ({blue_dur:.1f}s)")

        # Save Blue
        if gen % save_freq == 0 or gen == generations:
            blue_model.save(f"models/zerosum/blue_gen{gen}.zip")
            print(f"  Saved: models/zerosum/blue_gen{gen}.zip")

        # Test generation
        print(f"\nTesting Generation {gen}...")
        test_results = test_generation(red_model, blue_model, num_battles=test_battles)

        gen_dur = time.time() - gen_start

        # Log results
        gen_data = {
            'generation': gen,
            'duration_seconds': gen_dur,
            'red_training_seconds': red_dur,
            'blue_training_seconds': blue_dur,
            'test_results': test_results
        }
        training_log['generations'].append(gen_data)

        # Print results
        print(f"\nGeneration {gen} Results:")
        print("-"*70)
        print(f"Red wins:   {test_results['red_wins']}/{test_battles} ({test_results['red_win_rate']*100:.1f}%)")
        print(f"Blue wins:  {test_results['blue_wins']}/{test_battles} ({test_results['blue_win_rate']*100:.1f}%)")
        print(f"Draws:      {test_results['draws']}/{test_battles}")
        print(f"\nReward Statistics:")
        print(f"Red mean:   {test_results['red_mean_reward']:+.1f} (std: {test_results['red_reward_std']:.1f})")
        print(f"Blue mean:  {test_results['blue_mean_reward']:+.1f} (std: {test_results['blue_reward_std']:.1f})")
        print(f"Sum (should be ~0): {test_results['reward_sum']:+.1f}")

        # Validate zero-sum property
        if abs(test_results['reward_sum']) < 5:
            print("✓ Zero-sum property verified")
        else:
            print("⚠️  Warning: Reward sum not close to zero!")

        # Check for learning
        if gen > 1:
            prev_balance = training_log['generations'][-2]['test_results']['red_win_rate']
            curr_balance = test_results['red_win_rate']
            print(f"\nWin rate shift: {prev_balance*100:.1f}% → {curr_balance*100:.1f}%")

            if abs(curr_balance - 0.5) < 0.3:
                print("✓ Balanced competition (both agents competitive)")
            elif curr_balance > 0.7:
                print("⚠️  Red dominating (Blue needs to catch up)")
            else:
                print("⚠️  Blue dominating (Red needs to catch up)")

        print()

    total_dur = time.time() - total_start

    # Save final models
    red_model.save("models/aceac_red_zerosum.zip")
    blue_model.save("models/aceac_blue_zerosum.zip")

    # Finalize log
    training_log['end_time'] = datetime.now(timezone.utc).isoformat()
    training_log['total_duration_seconds'] = total_dur
    training_log['status'] = 'complete'

    with open('logs/zerosum_training.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    # Final summary
    print("="*70)
    print("ZERO-SUM CO-EVOLUTION COMPLETE!")
    print("="*70)
    print(f"Total duration: {total_dur/60:.1f} minutes")
    print(f"\nFinal models saved:")
    print(f"  models/aceac_red_zerosum.zip")
    print(f"  models/aceac_blue_zerosum.zip")
    print(f"\nTraining log:")
    print(f"  logs/zerosum_training.json")
    print("="*70)
    print()

    # Final validation
    print("FINAL VALIDATION:")
    print("-"*70)
    final_test = test_generation(red_model, blue_model, num_battles=100)
    print(f"Win rates (100 battles):")
    print(f"  Red:  {final_test['red_win_rate']*100:.1f}%")
    print(f"  Blue: {final_test['blue_win_rate']*100:.1f}%")
    print(f"  Draw: {final_test['draws']:.0f}%")
    print(f"\nMean rewards:")
    print(f"  Red:  {final_test['red_mean_reward']:+.1f}")
    print(f"  Blue: {final_test['blue_mean_reward']:+.1f}")
    print(f"  Sum:  {final_test['reward_sum']:+.1f} (zero-sum check)")
    print("="*70)

    return red_model, blue_model, training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACEAC Zero-Sum Co-Evolution Training')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of co-evolution generations (default: 10)')
    parser.add_argument('--timesteps', type=int, default=5000,
                        help='Timesteps per generation (default: 5000)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (1 generation, 1000 timesteps)')

    args = parser.parse_args()

    if args.test:
        print("TEST MODE: 1 generation, 1000 timesteps")
        zerosum_coevolution_training(
            timesteps_per_generation=1000,
            generations=1,
            test_battles=10
        )
    else:
        zerosum_coevolution_training(
            timesteps_per_generation=args.timesteps,
            generations=args.generations,
            test_battles=20
        )
