# Dynamic Co-Evolution System - Validation Guide

**Author:** @sarowarzahan414
**Date:** 2025-11-19
**Purpose:** Validate the fully dynamic co-evolution system works correctly

---

## Overview

This guide validates that the dynamic co-evolution system (with NO predefined logic) works correctly before running full training experiments.

**Key Validation Points:**
1. ✅ Environment initializes correctly
2. ✅ State dynamics work (actions affect state)
3. ✅ Opponent interaction functions
4. ✅ Diversity bonus mechanism works
5. ✅ Training loop completes
6. ✅ No hardcoded logic present

---

## Quick Validation

### Automated Test Suite

Run the validation script to test all core functionality:

```bash
python validate_dynamic_system.py
```

**Expected Output:**
```
======================================================================
DYNAMIC CO-EVOLUTION SYSTEM - VALIDATION SUITE
======================================================================

TEST 1: Environment Creation
✓ Environment created successfully
  - Action space: Discrete(25)
  - Observation space: Box(0.0, 1.0, (32,), float32)

TEST 2: Environment Reset
✓ Environment reset successful
  - Observation shape: (32,)
  - State mean: 0.xxx

... [more tests] ...

======================================================================
VALIDATION SUMMARY
======================================================================
Results: 7/7 tests passed
✅ ALL TESTS PASSED - System ready for full training!
```

---

## Manual Validation Steps

### Step 1: Verify No Predefined Logic

**Check:** Confirm actions have NO hardcoded meanings

```bash
grep -n "attack_power\s*=" aceac_dynamic_coevolution.py
grep -n "defense_power\s*=" aceac_dynamic_coevolution.py
grep -n "tool_phase_map" aceac_dynamic_coevolution.py
```

**Expected:** NO matches (these patterns shouldn't exist in dynamic system)

**Verify in code:**
```python
# aceac_dynamic_coevolution.py:153
def _dynamic_state_transition(self, action: int, opponent_action: Optional[int]):
    # Uses action as SEED, not hardcoded mapping
    np.random.seed(action)  # Deterministic but not predefined
    affected_dims = np.random.choice(...)  # Random dimensions

    # NO hardcoded "if action == 5: do_exploit()"
```

---

### Step 2: Test Environment Basics

**Interactive Python Test:**

```python
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv
import numpy as np

# Create environment
env = DynamicCoEvolutionEnv(agent_role="red", num_actions=25, state_dim=32)

# Reset and observe
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial state mean: {np.mean(env.state):.3f}")

# Take action and observe effect
action = 10
obs, reward, done, _, info = env.step(action)
print(f"After action {action}:")
print(f"  Reward: {reward:.3f}")
print(f"  State mean: {info['state_mean']:.3f}")
print(f"  State std: {info['state_std']:.3f}")

# Verify state changes with different actions
initial_state = env.state.copy()
env.step(5)
state_after_5 = env.state.copy()

env.reset()
env.step(15)
state_after_15 = env.state.copy()

# Different actions should affect state differently
diff_5 = np.mean(np.abs(state_after_5 - initial_state))
diff_15 = np.mean(np.abs(state_after_15 - initial_state))
print(f"\nState change from action 5: {diff_5:.4f}")
print(f"State change from action 15: {diff_15:.4f}")
print("✓ Different actions have different effects" if diff_5 != diff_15 else "⚠️ Issue detected")
```

---

### Step 3: Test Opponent Interaction

```python
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv
from stable_baselines3 import PPO

# Create opponent
opponent_env = DynamicCoEvolutionEnv(agent_role="blue")
opponent = PPO("MlpPolicy", opponent_env, verbose=0)

# Create env with opponent
env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=opponent)

# Run episode
obs, _ = env.reset()
for step in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)

    print(f"Step {step}: Red={action}, Blue={info['opponent_action']}, Reward={reward:.2f}")

    if done:
        break

# Verify opponent is actually playing
print("✓ Opponent interaction working" if info['opponent_action'] >= 0 else "⚠️ Opponent not responding")
```

---

### Step 4: Test Diversity Mechanism

```python
from aceac_dynamic_coevolution import DynamicCoEvolutionEnv

env = DynamicCoEvolutionEnv(agent_role="red")
obs, _ = env.reset()

# Test 1: Low diversity (same action)
for _ in range(10):
    env.step(5)  # Always action 5

low_diversity_bonus = env._diversity_bonus()
print(f"Low diversity bonus: {low_diversity_bonus:.3f}")

# Test 2: High diversity (different actions)
obs, _ = env.reset()
for i in range(10):
    env.step(i)  # Actions 0-9

high_diversity_bonus = env._diversity_bonus()
print(f"High diversity bonus: {high_diversity_bonus:.3f}")

# Verify diversity bonus works
if high_diversity_bonus > low_diversity_bonus:
    print("✓ Diversity bonus mechanism working correctly")
else:
    print("⚠️ Diversity bonus not working as expected")
```

---

### Step 5: Mini Training Run

```python
from aceac_dynamic_coevolution import DynamicCoEvolutionTrainer

# Quick test with reduced parameters
trainer = DynamicCoEvolutionTrainer(
    num_actions=10,   # Reduced for speed
    state_dim=16,     # Reduced for speed
    population_size=2
)

# Train for just 2 generations
red, blue = trainer.train(
    num_generations=2,
    steps_per_generation=1000,  # Very short
    save_dir="models/validation_test"
)

print("✓ Mini training completed successfully")
print(f"Generation history length: {len(trainer.generation_history)}")

# Check that metrics were recorded
assert len(trainer.generation_history) == 2
assert 'evaluation' in trainer.generation_history[0]
assert 'state_dominance' in trainer.generation_history[0]['evaluation']
print("✓ All metrics recorded correctly")
```

---

## Validation Checklist

Before running full training, verify:

### Environment Setup
- [ ] `DynamicCoEvolutionEnv` initializes without errors
- [ ] Environment reset produces valid observations
- [ ] Step function executes without errors
- [ ] State bounds maintained (values in [0, 1])

### Dynamic Behavior
- [ ] NO hardcoded action meanings in code
- [ ] NO predetermined attack/defense powers
- [ ] NO fixed tool-to-phase mappings
- [ ] Actions affect state deterministically but not semantically

### Mechanisms
- [ ] Diversity bonus increases with action variety
- [ ] State changes when actions are taken
- [ ] Different actions have different effects
- [ ] Opponent interaction works when opponent provided

### Training
- [ ] Trainer initializes successfully
- [ ] Training loop completes without crashes
- [ ] Models save correctly
- [ ] Training history recorded with all metrics

### Code Quality
- [ ] No import errors
- [ ] No syntax errors
- [ ] Gymnasium environment properly subclassed
- [ ] Stable-Baselines3 compatible

---

## Expected Behavior

### What SHOULD Happen:

✅ **Actions have random initial effects**
- Action 5 might affect dimensions [2, 7, 12, 18]
- Action 10 might affect dimensions [1, 9, 15, 22]
- Effects determined by seed, not hardcoded logic

✅ **Rewards based on outcomes**
- State moves favorably → positive reward
- State moves unfavorably → negative reward
- Diversity maintained → bonus reward

✅ **Co-evolution emerges**
- Red learns what works against Blue
- Blue learns to counter Red's strategies
- Neither has predetermined "correct" actions

### What SHOULD NOT Happen:

❌ **Hardcoded effectiveness values**
- NO `attack_power = 0.18`
- NO `defense_power = 0.12`
- NO predetermined action strengths

❌ **Predetermined sequences**
- NO "must do reconnaissance before exploit"
- NO "action 12 = SQL injection"
- NO semantic action meanings

❌ **Complex reward formulas**
- NO `reward = attack * 100 + stealth * 50 - detection * 75`
- Simple sparse rewards only

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution:**
```bash
pip install gymnasium stable-baselines3 numpy
```

---

### Issue: Environment Step Errors

**Error:**
```
TypeError: 'NoneType' object is not subscriptable
```

**Solution:**
Check that environment is properly initialized:
```python
env = DynamicCoEvolutionEnv(agent_role="red")  # Required parameter
obs, _ = env.reset()  # Must reset before stepping
```

---

### Issue: Opponent Model Errors

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'predict'
```

**Solution:**
Opponent model must be a trained SB3 model:
```python
# Create and train opponent first
opponent_env = DynamicCoEvolutionEnv(agent_role="blue")
opponent = PPO("MlpPolicy", opponent_env, verbose=0)
# opponent.learn(total_timesteps=1000)  # Optional: quick training

# Then use in environment
env = DynamicCoEvolutionEnv(agent_role="red", opponent_model=opponent)
```

---

### Issue: State Values Out of Bounds

**Error:**
```
State values > 1.0 or < 0.0
```

**Solution:**
This shouldn't happen - state is clipped in `_dynamic_state_transition()`:
```python
new_state = np.clip(new_state, 0.0, 1.0)
```

If it occurs, check for bugs in state transition logic.

---

### Issue: No State Changes

**Symptom:**
State remains constant despite actions

**Check:**
```python
# Verify state actually changes
initial = env.state.copy()
env.step(5)
final = env.state.copy()
change = np.mean(np.abs(final - initial))
print(f"State change: {change}")  # Should be > 0.001
```

**Possible causes:**
- Effect size too small
- Noise canceling out effects
- Bug in state transition

---

## Performance Benchmarks

### Validation Test Suite

**Hardware:** Standard CPU
**Expected Runtime:**
- Test 1-6: < 5 seconds total
- Test 7 (Mini Training): 1-2 minutes

**If slower:**
- Reduce `steps_per_generation` in Test 7
- Check CPU usage
- Ensure no GPU errors (can run on CPU)

---

### Mini Training Run

**Configuration:**
- 2 generations
- 1000 steps per generation
- 10 actions, 16 state dims

**Expected Runtime:** 1-3 minutes

**Memory Usage:** < 500 MB

---

## Integration with Full System

### After Validation Passes:

1. **Run full training:**
   ```bash
   python aceac_dynamic_coevolution.py
   ```

2. **Monitor with edge case detector:**
   ```bash
   python coevolution_edge_case_detector.py \
       --training-log models/dynamic/training_history.json
   ```

3. **Analyze results:**
   - Check `models/dynamic/training_history.json`
   - Verify state dominance oscillates (co-evolution)
   - Confirm action diversity > 0.3
   - Ensure no edge cases detected

4. **Compare to static system:**
   - Review `DYNAMIC_VS_STATIC_COMPARISON.md`
   - Analyze emergent vs hardcoded strategies
   - Document novel behaviors discovered

---

## Success Criteria

### Validation Passes If:

✅ All 7 automated tests pass
✅ No hardcoded logic found in code inspection
✅ State changes with actions
✅ Diversity mechanism functions
✅ Opponent interaction works
✅ Mini training completes
✅ Metrics recorded correctly

### Ready for Full Training When:

✅ Validation suite passes
✅ No errors in mini training
✅ Memory usage reasonable
✅ Code follows dynamic philosophy

---

## Next Steps After Validation

1. **Full Training Run:**
   - 20 generations
   - 50,000 steps per generation
   - Save to `models/dynamic/`

2. **Analysis:**
   - Load `training_history.json`
   - Visualize state dominance over time
   - Analyze action diversity trends
   - Identify emergent strategies

3. **Edge Case Detection:**
   - Run `coevolution_edge_case_detector.py`
   - Review health score
   - Address any critical issues

4. **Documentation:**
   - Document emergent behaviors
   - Compare to static baseline
   - Record novel strategies discovered

---

## Files

**Validation:**
- `validate_dynamic_system.py` - Automated test suite
- `VALIDATION_GUIDE.md` - This file

**System:**
- `aceac_dynamic_coevolution.py` - Main implementation
- `README_DYNAMIC_SYSTEM.md` - Usage guide
- `DYNAMIC_VS_STATIC_COMPARISON.md` - Philosophy comparison

**Monitoring:**
- `coevolution_edge_case_detector.py` - Edge case detection
- `test_coevolution_edge_cases.py` - Edge case tests

---

**Version:** 1.0
**Last Updated:** 2025-11-19
**Status:** Ready for Use

**Validate Before Training - Ensure Pure Learning!** 🧠
