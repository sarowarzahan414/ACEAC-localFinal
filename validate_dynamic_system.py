"""
Quick validation script for dynamic co-evolution system

Tests basic functionality without long training runs.
Ensures the system works correctly before full training.

Author: @sarowarzahan414
Date: 2025-11-19
"""

import sys
import numpy as np

try:
    from aceac_dynamic_coevolution import DynamicCoEvolutionEnv, DynamicCoEvolutionTrainer
    from stable_baselines3 import PPO
    print("✓ Imports successful\n")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please install dependencies: pip install gymnasium stable-baselines3 numpy")
    sys.exit(1)


def test_environment_creation():
    """Test basic environment initialization"""
    print("="*70)
    print("TEST 1: Environment Creation")
    print("="*70)

    try:
        env = DynamicCoEvolutionEnv(
            agent_role="red",
            opponent_model=None,
            num_actions=25,
            state_dim=32
        )
        print(f"✓ Environment created successfully")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Max steps: {env.max_steps}")
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False


def test_environment_reset():
    """Test environment reset functionality"""
    print("\n" + "="*70)
    print("TEST 2: Environment Reset")
    print("="*70)

    try:
        env = DynamicCoEvolutionEnv(agent_role="red")
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  - State mean: {np.mean(env.state):.3f}")
        return True
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return False


def test_environment_step():
    """Test environment stepping with random actions"""
    print("\n" + "="*70)
    print("TEST 3: Environment Step (No Opponent)")
    print("="*70)

    try:
        env = DynamicCoEvolutionEnv(agent_role="red")
        obs, _ = env.reset()

        # Take 10 random steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if i == 0:
                print(f"✓ First step successful")
                print(f"  - Action: {action}")
                print(f"  - Reward: {reward:.3f}")
                print(f"  - State mean: {info['state_mean']:.3f}")

        print(f"✓ Completed 10 steps without errors")
        print(f"  - Final state mean: {info['state_mean']:.3f}")
        print(f"  - Final state std: {info['state_std']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        return False


def test_state_dynamics():
    """Test that state actually changes with actions"""
    print("\n" + "="*70)
    print("TEST 4: State Dynamics (Actions Have Effect)")
    print("="*70)

    try:
        env = DynamicCoEvolutionEnv(agent_role="red")
        obs, _ = env.reset()
        initial_state = env.state.copy()

        # Take action
        action = 5
        obs, reward, _, _, info = env.step(action)
        new_state = env.state.copy()

        # Check state changed
        state_diff = np.mean(np.abs(new_state - initial_state))

        print(f"✓ State dynamics working")
        print(f"  - Initial state mean: {np.mean(initial_state):.3f}")
        print(f"  - New state mean: {np.mean(new_state):.3f}")
        print(f"  - State change magnitude: {state_diff:.3f}")

        if state_diff > 0.001:
            print(f"✓ State changes detected (good!)")
            return True
        else:
            print(f"⚠️  State change very small - might be an issue")
            return False

    except Exception as e:
        print(f"❌ State dynamics test failed: {e}")
        return False


def test_opponent_interaction():
    """Test environment with opponent model"""
    print("\n" + "="*70)
    print("TEST 5: Opponent Interaction")
    print("="*70)

    try:
        # Create a dummy opponent
        dummy_env = DynamicCoEvolutionEnv(agent_role="blue")
        dummy_opponent = PPO("MlpPolicy", dummy_env, verbose=0)

        # Create env with opponent
        env = DynamicCoEvolutionEnv(
            agent_role="red",
            opponent_model=dummy_opponent
        )
        obs, _ = env.reset()

        # Take step
        action = 5
        obs, reward, _, _, info = env.step(action)

        print(f"✓ Opponent interaction working")
        print(f"  - Red action: {action}")
        print(f"  - Blue action: {info['opponent_action']}")
        print(f"  - Reward: {reward:.3f}")

        if info['opponent_action'] >= 0:
            print(f"✓ Opponent provided valid action")
            return True
        else:
            print(f"⚠️  Opponent action invalid")
            return False

    except Exception as e:
        print(f"❌ Opponent interaction test failed: {e}")
        return False


def test_diversity_bonus():
    """Test that diversity bonus works"""
    print("\n" + "="*70)
    print("TEST 6: Diversity Bonus Mechanism")
    print("="*70)

    try:
        env = DynamicCoEvolutionEnv(agent_role="red")
        obs, _ = env.reset()

        # Take same action 10 times (low diversity)
        for _ in range(10):
            obs, reward_same, _, _, _ = env.step(5)

        diversity_bonus_same = env._diversity_bonus()

        # Reset and take different actions (high diversity)
        obs, _ = env.reset()
        for i in range(10):
            obs, reward_diff, _, _, _ = env.step(i)

        diversity_bonus_diff = env._diversity_bonus()

        print(f"✓ Diversity bonus mechanism working")
        print(f"  - Low diversity bonus: {diversity_bonus_same:.3f}")
        print(f"  - High diversity bonus: {diversity_bonus_diff:.3f}")

        if diversity_bonus_diff > diversity_bonus_same:
            print(f"✓ Diversity bonus higher for diverse actions (correct!)")
            return True
        else:
            print(f"⚠️  Diversity bonus not working as expected")
            return False

    except Exception as e:
        print(f"❌ Diversity bonus test failed: {e}")
        return False


def test_mini_training():
    """Test minimal training run"""
    print("\n" + "="*70)
    print("TEST 7: Mini Training Run (2 generations, 1000 steps each)")
    print("="*70)

    try:
        trainer = DynamicCoEvolutionTrainer(
            num_actions=10,  # Reduced for speed
            state_dim=16,    # Reduced for speed
            population_size=2
        )

        print("Starting mini training...")
        red_model, blue_model = trainer.train(
            num_generations=2,
            steps_per_generation=1000,
            save_dir="models/validation_test"
        )

        print(f"\n✓ Mini training completed successfully")
        print(f"  - Red model: {red_model}")
        print(f"  - Blue model: {blue_model}")
        print(f"  - Generations: {len(trainer.generation_history)}")

        return True

    except Exception as e:
        print(f"❌ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("DYNAMIC CO-EVOLUTION SYSTEM - VALIDATION SUITE")
    print("="*70)
    print("Testing basic functionality before full training\n")

    tests = [
        ("Environment Creation", test_environment_creation),
        ("Environment Reset", test_environment_reset),
        ("Environment Step", test_environment_step),
        ("State Dynamics", test_state_dynamics),
        ("Opponent Interaction", test_opponent_interaction),
        ("Diversity Bonus", test_diversity_bonus),
        ("Mini Training", test_mini_training),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} threw exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print(f"✅ ALL TESTS PASSED - System ready for full training!")
    else:
        print(f"⚠️  {total - passed} test(s) failed - review errors above")

    print(f"{'='*70}\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
