import numpy as np
from aceac_zerosum_environment import ZeroSumCyberEnv

# Test if SIMPLE strategy can win
env = ZeroSumCyberEnv(agent_role="red", opponent_model=None)
obs, _ = env.reset()

red_wins = 0
blue_wins = 0
draws = 0

for episode in range(100):
    obs, _ = env.reset()

    for step in range(100):
        # Red: Always use action 0
        # Blue: No opponent (None)

        red_action = 0  # Consistent strategy
        obs, reward, done, truncated, info = env.step(red_action)

        if done or truncated:
            if info['winner'] == 'red':
                red_wins += 1
            elif info['winner'] == 'blue':
                blue_wins += 1
            else:
                draws += 1
            break

print(f"Simple Strategy Results (100 episodes):")
print(f"  Red wins: {red_wins}%")
print(f"  Blue wins: {blue_wins}%")
print(f"  Draws: {draws}%")
print()

# Assess results
if red_wins > 80:
    print("✓ PASS: Red can win consistently without opponent")
elif red_wins > 50:
    print("⚠️  MARGINAL: Red wins sometimes but inconsistently")
elif draws > 80:
    print("❌ FAIL: Environment produces mostly draws (saturation bug?)")
else:
    print("❌ FAIL: Red cannot win (environment broken)")
