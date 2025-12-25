"""
Diagnostic script to test Blue environment
"""
from aceac_v2_cyber_killchain import ACEACv2Environment
import numpy as np

print("Testing Blue environment:")
env = ACEACv2Environment(agent_role="blue")
obs, _ = env.reset()

total_reward = 0.0
successes = 0

for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if reward > 0:
        successes += 1
    print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}")

print(f"\nTotal reward: {total_reward:.2f}")
print(f"Successes: {successes}/10")
print(f"Blue environment working: {'YES' if successes > 0 else 'NO - ALL DEFENSES FAILED'}")
