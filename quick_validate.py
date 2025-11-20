"""
Quick validation of trained agents vs baselines
"""

import numpy as np
from stable_baselines3 import PPO
from aceac_zerosum_environment import ZeroSumCyberEnv

# Load trained models
print("Loading trained models...")
try:
    red_model = PPO.load('models/aceac_red_zerosum.zip')
    blue_model = PPO.load('models/aceac_blue_zerosum.zip')
    print("✓ Models loaded")
except:
    print("❌ Could not load models from models/aceac_red_zerosum.zip")
    print("   Make sure you ran training first!")
    exit(1)

# Simple baseline agents
class RandomAgent:
    """Random baseline"""
    def __init__(self):
        self.name = "Random"
    def predict(self, obs, deterministic=True):
        return np.random.randint(0, 25), None

class GreedyAgent:
    """Simple heuristic baseline"""
    def __init__(self, role="red"):
        self.role = role
        self.name = f"Greedy-{role}"
    def predict(self, obs, deterministic=True):
        state_mean = np.mean(obs)
        if self.role == "red":
            if state_mean < 0.5:
                return np.random.randint(15, 25), None
            else:
                return np.random.randint(5, 15), None
        else:
            if state_mean > 0.5:
                return np.random.randint(15, 25), None
            else:
                return np.random.randint(5, 15), None

def test_vs_baseline(trained_model, trained_role, baseline_agent, baseline_role, num_episodes=50):
    """Test trained agent vs baseline"""
    env = ZeroSumCyberEnv(agent_role=trained_role, opponent_model=baseline_agent)

    trained_wins = 0
    baseline_wins = 0
    draws = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        for step in range(100):
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                winner = info['winner']
                if winner == trained_role:
                    trained_wins += 1
                elif winner == baseline_role:
                    baseline_wins += 1
                else:
                    draws += 1
                break

    return trained_wins, baseline_wins, draws

# Run tests
print("\n" + "="*70)
print("BASELINE VALIDATION TESTS")
print("="*70)

# Test 1: Red vs Random Blue
print("\nTest 1: Trained Red vs Random Blue (50 episodes)")
r_wins, b_wins, draws = test_vs_baseline(red_model, "red", RandomAgent(), "blue", 50)
print(f"  Trained Red wins: {r_wins}/50 ({r_wins/50*100:.1f}%)")
print(f"  Random Blue wins: {b_wins}/50 ({b_wins/50*100:.1f}%)")
print(f"  Draws: {draws}/50 ({draws/50*100:.1f}%)")

if r_wins > 35:
    print("  ✅ PASS - Beats random significantly")
elif r_wins > 25:
    print("  ⚠️  MARGINAL - Beats random but not strongly")
else:
    print("  ❌ FAIL - Doesn't beat random")

# Test 2: Red vs Greedy Blue
print("\nTest 2: Trained Red vs Greedy Blue (50 episodes)")
r_wins, b_wins, draws = test_vs_baseline(red_model, "red", GreedyAgent("blue"), "blue", 50)
print(f"  Trained Red wins: {r_wins}/50 ({r_wins/50*100:.1f}%)")
print(f"  Greedy Blue wins: {b_wins}/50 ({b_wins/50*100:.1f}%)")
print(f"  Draws: {draws}/50 ({draws/50*100:.1f}%)")

if r_wins > 30:
    print("  ✅ PASS - Beats greedy heuristic")
elif r_wins > 20:
    print("  ⚠️  MARGINAL - Competitive with greedy")
else:
    print("  ❌ FAIL - Loses to simple heuristic")

# Test 3: Blue vs Random Red
print("\nTest 3: Trained Blue vs Random Red (50 episodes)")
b_wins, r_wins, draws = test_vs_baseline(blue_model, "blue", RandomAgent(), "red", 50)
print(f"  Trained Blue wins: {b_wins}/50 ({b_wins/50*100:.1f}%)")
print(f"  Random Red wins: {r_wins}/50 ({r_wins/50*100:.1f}%)")
print(f"  Draws: {draws}/50 ({draws/50*100:.1f}%)")

if b_wins > 35:
    print("  ✅ PASS - Beats random significantly")
elif b_wins > 25:
    print("  ⚠️  MARGINAL - Beats random but not strongly")
else:
    print("  ❌ FAIL - Doesn't beat random")

# Test 4: Blue vs Greedy Red
print("\nTest 4: Trained Blue vs Greedy Red (50 episodes)")
b_wins, r_wins, draws = test_vs_baseline(blue_model, "blue", GreedyAgent("red"), "red", 50)
print(f"  Trained Blue wins: {b_wins}/50 ({b_wins/50*100:.1f}%)")
print(f"  Greedy Red wins: {r_wins}/50 ({r_wins/50*100:.1f}%)")
print(f"  Draws: {draws}/50 ({draws/50*100:.1f}%)")

if b_wins > 30:
    print("  ✅ PASS - Beats greedy heuristic")
elif b_wins > 20:
    print("  ⚠️  MARGINAL - Competitive with greedy")
else:
    print("  ❌ FAIL - Loses to simple heuristic")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print("\nInterpretation:")
print("  - Gen 1 with only 1000 timesteps is very early training")
print("  - High draw rates are EXPECTED at this stage")
print("  - If agents beat random at all, that is learning")
print("  - Full training (20 gen, 5000 steps/gen) should improve significantly")
