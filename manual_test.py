#!/usr/bin/env python3
"""
Manual Testing Script - Verify Core Mechanics Work

Tests fundamental behavior BEFORE any complex training:
A. Reward sanity checks
B. State dynamics verification

Run this on Kali Linux:
    python3 manual_test.py

Author: @sarowarzahan414
Date: 2025-11-19
"""

import numpy as np
import sys

try:
    from aceac_dynamic_coevolution import DynamicCoEvolutionEnv
    print("✓ Successfully imported DynamicCoEvolutionEnv\n")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print("\nInstall dependencies first:")
    print("  pip3 install gymnasium stable-baselines3 numpy")
    sys.exit(1)


print("="*70)
print("MANUAL TEST - CORE MECHANICS VERIFICATION")
print("="*70)
print()


# TEST A: REWARD SANITY
print("="*70)
print("TEST A: REWARD SANITY")
print("="*70)
print()

print("A1: Red attacks, Blue does nothing")
print("-" * 70)

# Create Red environment (no opponent)
red_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
obs, _ = red_env.reset()

print(f"Initial state mean: {np.mean(red_env.state):.4f}")

# Red takes 10 actions
red_rewards = []
for i in range(10):
    action = red_env.action_space.sample()
    obs, reward, done, _, info = red_env.step(action)
    red_rewards.append(reward)

    if i < 3:  # Print first 3 steps
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, state_mean={info['state_mean']:.4f}")

print(f"\nRed agent (attacking, no opponent):")
print(f"  Total rewards: {red_rewards}")
print(f"  Mean reward: {np.mean(red_rewards):.3f}")
print(f"  Final state mean: {np.mean(red_env.state):.4f}")

if np.mean(red_rewards) > 0:
    print("  ✓ GOOD: Red gets positive reward on average when attacking freely")
else:
    print("  ⚠️  PROBLEM: Red should get positive reward when attacking freely!")

print()

print("A2: Blue defends, Red does nothing")
print("-" * 70)

# Create Blue environment (no opponent)
blue_env = DynamicCoEvolutionEnv(agent_role="blue", opponent_model=None)
obs, _ = blue_env.reset()

print(f"Initial state mean: {np.mean(blue_env.state):.4f}")

# Blue takes 10 actions
blue_rewards = []
for i in range(10):
    action = blue_env.action_space.sample()
    obs, reward, done, _, info = blue_env.step(action)
    blue_rewards.append(reward)

    if i < 3:  # Print first 3 steps
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, state_mean={info['state_mean']:.4f}")

print(f"\nBlue agent (defending, no opponent):")
print(f"  Total rewards: {blue_rewards}")
print(f"  Mean reward: {np.mean(blue_rewards):.3f}")
print(f"  Final state mean: {np.mean(blue_env.state):.4f}")

if np.mean(blue_rewards) > 0:
    print("  ✓ GOOD: Blue gets positive reward on average when defending freely")
else:
    print("  ⚠️  PROBLEM: Blue should get positive reward when defending freely!")

print()

print("A3: Both do nothing (or minimal actions)")
print("-" * 70)

# Test with same action repeatedly (minimal change)
minimal_env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
obs, _ = minimal_env.reset()

print(f"Initial state mean: {np.mean(minimal_env.state):.4f}")

# Repeat same action
minimal_rewards = []
same_action = 5  # Arbitrary action
for i in range(10):
    obs, reward, done, _, info = minimal_env.step(same_action)
    minimal_rewards.append(reward)

    if i < 3:
        print(f"  Step {i+1}: action={same_action}, reward={reward:.3f}, state_mean={info['state_mean']:.4f}")

print(f"\nMinimal activity (same action repeated):")
print(f"  Mean reward: {np.mean(minimal_rewards):.3f}")
print(f"  Final state mean: {np.mean(minimal_env.state):.4f}")

if abs(np.mean(minimal_rewards)) < 10.0:
    print("  ✓ GOOD: Minimal activity gives small rewards")
else:
    print("  ⚠️  UNUSUAL: Expected small rewards for minimal activity")

print()
print()


# TEST B: STATE DYNAMICS
print("="*70)
print("TEST B: STATE DYNAMICS")
print("="*70)
print()

print("B1: Run 10 episodes, check if state changes")
print("-" * 70)

env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)

episode_states = []
episode_rewards = []

for episode in range(10):
    obs, _ = env.reset()
    initial_state = env.state.copy()

    # Run full episode
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        if done:
            break

    final_state = env.state.copy()
    state_change = np.mean(np.abs(final_state - initial_state))

    episode_states.append({
        'initial_mean': np.mean(initial_state),
        'final_mean': np.mean(final_state),
        'change': state_change
    })
    episode_rewards.append(total_reward)

    print(f"  Episode {episode+1}: initial={np.mean(initial_state):.4f}, "
          f"final={np.mean(final_state):.4f}, "
          f"change={state_change:.4f}, "
          f"reward={total_reward:.2f}")

print(f"\nState change statistics:")
changes = [ep['change'] for ep in episode_states]
print(f"  Mean state change: {np.mean(changes):.4f}")
print(f"  Std state change: {np.std(changes):.4f}")
print(f"  Min change: {np.min(changes):.4f}")
print(f"  Max change: {np.max(changes):.4f}")

if np.mean(changes) > 0.01:
    print("  ✓ GOOD: State changes significantly with actions")
else:
    print("  ⚠️  PROBLEM: State barely changes! Actions may have no effect.")

print()

print("B2: Verify state isn't just random noise")
print("-" * 70)

# Take same action sequence twice - should get similar results
env1 = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
env2 = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)

# Set same seed
env1.reset(seed=42)
env2.reset(seed=42)

action_sequence = [3, 7, 12, 5, 18, 9, 14, 2, 11, 6]

states1 = [env1.state.copy()]
states2 = [env2.state.copy()]

for action in action_sequence:
    env1.step(action)
    env2.step(action)
    states1.append(env1.state.copy())
    states2.append(env2.state.copy())

# Compare final states
final_diff = np.mean(np.abs(states1[-1] - states2[-1]))

print(f"Same seed, same actions:")
print(f"  Environment 1 final state mean: {np.mean(states1[-1]):.6f}")
print(f"  Environment 2 final state mean: {np.mean(states2[-1]):.6f}")
print(f"  Difference: {final_diff:.6f}")

if final_diff < 0.0001:
    print("  ✓ GOOD: Deterministic behavior (not random noise)")
else:
    print("  ⚠️  PROBLEM: State evolution is random! Should be deterministic.")

print()

print("B3: Different actions should produce different states")
print("-" * 70)

# Test that different actions lead to different outcomes
env_a = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)
env_b = DynamicCoEvolutionEnv(agent_role="red", opponent_model=None)

env_a.reset(seed=42)
env_b.reset(seed=42)

# Same initial state, different actions
for i in range(10):
    env_a.step(5)   # Always action 5
    env_b.step(10)  # Always action 10

final_a = env_a.state.copy()
final_b = env_b.state.copy()

diff = np.mean(np.abs(final_a - final_b))

print(f"Same initial state, different actions:")
print(f"  Action 5 repeated → state mean: {np.mean(final_a):.4f}")
print(f"  Action 10 repeated → state mean: {np.mean(final_b):.4f}")
print(f"  Difference: {diff:.4f}")

if diff > 0.01:
    print("  ✓ GOOD: Different actions produce different states")
else:
    print("  ⚠️  PROBLEM: Different actions produce same states! No learning signal.")

print()
print()


# SUMMARY
print("="*70)
print("SUMMARY")
print("="*70)
print()

issues = []

# Check A1
if np.mean(red_rewards) <= 0:
    issues.append("A1: Red doesn't get positive reward when attacking")

# Check A2
if np.mean(blue_rewards) <= 0:
    issues.append("A2: Blue doesn't get positive reward when defending")

# Check B1
if np.mean(changes) <= 0.01:
    issues.append("B1: State barely changes with actions")

# Check B2
if final_diff >= 0.0001:
    issues.append("B2: State evolution is not deterministic")

# Check B3
if diff <= 0.01:
    issues.append("B3: Different actions don't produce different states")

if not issues:
    print("✅ ALL CORE MECHANICS TESTS PASSED")
    print()
    print("System is ready for validation suite:")
    print("  ./run_all_checks.sh")
    print()
else:
    print("❌ CRITICAL ISSUES FOUND:")
    print()
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("DO NOT PROCEED TO FULL VALIDATION UNTIL THESE ARE FIXED!")
    print()

print("="*70)
