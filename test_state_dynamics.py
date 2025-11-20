import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv

env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
obs, _ = env.reset()

print("Testing if actions actually move state...\n")

# Test each action's effect
for action in [0, 6, 12, 18, 24]:
    env.reset()
    initial_state = np.mean(env.state)

    state_trajectory = [initial_state]

    for step in range(20):
        obs, reward, done, truncated, info = env.step(action)
        state_trajectory.append(info['state_mean'])

        if done or truncated:
            break

    final_state = state_trajectory[-1]
    total_movement = final_state - initial_state

    print(f"Action {action:2d}: {initial_state:.3f} -> {final_state:.3f} (Δ = {total_movement:+.3f})")

# Expected: Different actions should produce different state movements
# Higher actions should have stronger effects (due to action_strength multiplier)
print()
print("Expected: Higher-numbered actions should have stronger effects")
print("(Action 0 has 0.75x strength, Action 24 has 1.25x strength)")
