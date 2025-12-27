# ACEAC v2.0 QD-SWAP RL - Complete Solution

## Problem Statement

The Blue agent in ACEAC v2.0 was stuck with:
- **Constant performance** (-800.0 across all 10 generations)
- **Zero behavioral diversity** (tool diversity = 0.000, behavior stuck at (0.043, 0.000, 1.000))
- **No policy variation** (only 1 policy in archive instead of diverse strategies)

## Root Causes Identified

1. **Defense power too weak** (0.08-0.12): Max defense (0.156) < Min threshold (0.18), making success mathematically impossible
2. **Sparse reward structure**: Simple +15/-8 rewards didn't encourage diverse behaviors
3. **Poor behavioral dimensions**: Using security gain as effectiveness metric hit ceiling
4. **Weak exploration**: ent_coef=0.1 caused early convergence to single tool exploitation
5. **Missing behavioral tracking**: No detection rate, false positive, or diversity metrics

## Complete Solution Implemented

### 1. Environment Updates (`aceac_v2_cyber_killchain.py`)

#### A. Balanced Defense Power
```python
# BEFORE: 10× too strong (0.80-1.20), then tried 5× (0.40-0.60)
# AFTER: 2.5× original (0.20-0.30) - BALANCED
tool_effectiveness = {
    **{i: 0.20 for i in range(5)},      # Network
    **{i: 0.30 for i in range(5, 10)},  # Firewall
    **{i: 0.25 for i in range(10, 15)}, # SIEM
    **{i: 0.22 for i in range(15, 20)}, # Endpoint
    **{i: 0.28 for i in range(20, 25)}, # IR
}
```

#### B. Attack/Defense Tracking
```python
# Added to __init__ and reset():
self.attacks_detected = 0
self.total_attacks = 0
self.false_positives = 0
self.defensive_actions = 0
self.recent_actions = []  # Last 25 actions for diversity
```

#### C. Dense Reward Shaping
```python
# Detection rewards
if attack_occurring:
    if detected:
        detection_reward = +10.0  # Caught attack
        time_bonus = 5.0 * (1.0 - response_time / 10.0)  # Faster = better
    else:
        detection_reward = -10.0  # Missed attack (false negative)
else:
    if raised_alarm:
        detection_reward = -5.0  # False positive penalty
    else:
        detection_reward = +2.0  # True negative (correct quiet)

# Diversity bonus
unique_tools = len(set(self.recent_actions))
diversity_bonus = 5.0 * (unique_tools / 25.0)

# Total reward
reward = detection_reward + time_bonus + diversity_bonus - cost_penalty
```

### 2. Training Updates (`aceac_v2_qd_swap_rl.py`)

#### A. MAP-Elites Archive
```python
class QDArchive:
    """MAP-Elites Archive for Quality-Diversity"""
    - 20x20x20 grid = 8,000 cells
    - Stores (policy, performance, behavior, generation) per cell
    - Discretizes continuous behavior [0,1]³ to grid coordinates
    - Updates cell only if new policy performs better
```

#### B. Superior Blue Behavioral Dimensions (User's Approach)

**Original Proposal:**
1. Detection Rate (attacks_detected / total_attacks)
2. Response Rate (defensive_actions / total_timesteps)
3. Resource Efficiency (reward / tools_used)

**Implementation with Fix:**
```python
# Dimension 1: Detection Rate (attacks caught / total attacks)
detection_rate = attacks_detected / max(total_attacks, 1)

# Dimension 2: Tool Switching Rate (FIXED from response_rate)
# Original response_rate was always 1.0 because RL agents act every step
# Replaced with tool switching to measure strategic adaptability
switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
switching_rate = switches / (len(actions) - 1)

# Dimension 3: Resource Efficiency (reward per unique tool)
efficiency = total_reward / max(unique_tools, 1)
efficiency_normalized = np.clip(efficiency / 50.0, 0.0, 1.0)
```

**Why Tool Switching Rate?**
- **Low switching (0.0-0.3)**: Specialist strategy (sticks to few tools)
- **Medium switching (0.4-0.6)**: Balanced approach (switches occasionally)
- **High switching (0.7-1.0)**: Adaptive strategy (constantly changes tools)

#### C. Strong Exploration
```python
# BEFORE: ent_coef=0.1 (weak exploration)
# AFTER: ent_coef=0.5 (strong exploration)
red_policy = PPO("MlpPolicy", env, ent_coef=0.5, ...)
blue_policy = PPO("MlpPolicy", env, ent_coef=0.5, ...)
```

## Response Rate Issue - Root Cause & Fix

### The Bug
```python
# Original (ALWAYS 1.0):
response_rate = defensive_actions / total_timesteps
# Problem: In Gym RL, agents MUST act every timestep
# Therefore: defensive_actions == total_timesteps ALWAYS
# Result: response_rate = 1.0 for ALL policies (no diversity)
```

### The Fix
```python
# Replaced with Tool Switching Rate:
switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
switching_rate = switches / (len(actions) - 1)
# This actually varies: specialist (low) vs adaptive (high) strategies
```

## Expected Results

Based on the previous successful run (from conversation summary):

### Before (Old System)
- Blue policies: **1** (stuck)
- Coverage: **0.01%** of archive
- Behavior: **(0.043, 0.000, 1.000)** - constant
- Performance: **-800.0** - constant

### After (Complete Solution)
- Blue policies: **15** diverse strategies
- Coverage: **0.19%** of 8,000 cells
- Behavior variation:
  - Detection rates: **0.367 - 0.900** ✓
  - Switching rates: **Expected to vary 0.0 - 1.0** ✓
  - Efficiency: **0.010 - 0.243** ✓
- Performance: **-800.0 → positive values** ✓

## Files Modified

1. **`aceac_v2_cyber_killchain.py`**
   - Lines 137-147: Added attack/defense tracking variables
   - Lines 167-172: Reset tracking metrics
   - Lines 307-392: Complete rewrite of `_execute_defensive_action()` with dense rewards

2. **`aceac_v2_qd_swap_rl.py`** (NEW FILE)
   - Complete implementation of QD-SWAP RL
   - MAP-Elites archive (lines 19-82)
   - QDSWAPTrainer with behavioral characterization (lines 85-320)
   - Superior Blue dimensions with tool switching fix (lines 173-242)
   - Main training function with strong exploration (lines 323-406)

3. **`test_qd_complete.py`** (NEW FILE)
   - Test suite for dense rewards, behavioral dimensions, and MAP-Elites

## How to Run

```bash
# Test the complete solution
python test_qd_complete.py

# Run full QD-SWAP RL training (20 generations, 20x20x20 grid)
python aceac_v2_qd_swap_rl.py

# Expected runtime: ~7-10 minutes for 20 generations
# Output: models/aceac_v2_qd/, logs/aceac_v2_qd_swap_rl.json
```

## Innovation Summary

This implementation includes:
- ✅ **Innovation #1**: QD Policy Pool with MAP-Elites (20x20x20 = 8,000 cells)
- ✅ **Innovation #2**: 3D Behavioral Characterization (detection, switching, efficiency)
- ✅ **User's Superior Approach**: Detection rate + efficiency metrics
- ✅ **Dense Reward Shaping**: Detection, FP penalties, response time, diversity
- ✅ **Strong Exploration**: ent_coef=0.5 for policy diversity
- ✅ **Balanced Defense**: 2.5× original power (0.20-0.30)
- ✅ **Tool Switching Fix**: Replaced constant 1.0 response_rate with varying switching_rate

## Next Steps

1. ✅ Run `test_qd_complete.py` to verify all components work
2. ✅ Commit changes with clear message
3. ⏭️ Run full training: `python aceac_v2_qd_swap_rl.py`
4. ⏭️ Analyze results and compare with baseline
5. ⏭️ Potentially move to multi-host environment (future work)

---

**Author**: @sarowarzahan414
**Date**: 2025-12-27
**Status**: Complete Solution Ready for Testing
