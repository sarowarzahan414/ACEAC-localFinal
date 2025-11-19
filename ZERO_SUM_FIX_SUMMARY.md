# Zero-Sum Reward System Fix - Summary

## Problem Identified (Brutal Analysis)

**Old System (BROKEN):**
- Red reward: `attack_power * 100` → Always POSITIVE
- Blue reward: `defense_power * 100` → Always POSITIVE
- **Result:** Both agents get participation trophies, no competition
- **This is COOPERATIVE, not ADVERSARIAL**

**Evidence:**
```
Red acting alone: +2.180 mean reward
Blue acting alone: +2.608 mean reward
Both doing minimal work: +1.227 mean reward
```

All positive! This means both can "win" simultaneously - fundamentally broken.

---

## Solution Implemented

### 1. Created `aceac_zerosum_environment.py`

**NEW System (FIXED):**
- **Win: +100**
- **Loss: -100**
- **Draw: 0**
- Small shaping bonus during episode (max ±0.25 per step)

**Win Conditions:**
- Red wins: State mean > 0.7 (network compromised)
- Blue wins: State mean < 0.3 (network secured) OR time runs out
- Draw: State mean between 0.3-0.7

**Zero-Sum Property:**
- Red's gain = Blue's loss
- Red: +100 → Blue: -100
- Rewards sum to ~0

### 2. Created `aceac_zerosum_training.py`

**Key Features:**
- Uses `ZeroSumCyberEnv` instead of broken `ACEACCoEvolutionEnv`
- Trains fresh agents (doesn't load broken reward models)
- Validates zero-sum property after each generation
- Tracks win rates (should be ~50/50 for balanced agents)
- Command-line arguments for testing

---

## How To Use - NEW COMMAND LINE

### Quick Test (1 generation, ~2-5 minutes):
```bash
python aceac_zerosum_training.py --test
```

### Full Training (10 generations, ~30-60 minutes):
```bash
python aceac_zerosum_training.py --generations 10
```

### Custom Training:
```bash
python aceac_zerosum_training.py --generations 20 --timesteps 10000
```

---

## Validation Tests

### Before Training:

**Compare Old vs New Systems:**
```bash
python compare_reward_systems.py
```

**Test Zero-Sum Environment:**
```bash
python aceac_zerosum_environment.py
```

Expected output:
- ✓ Red victory → Red: +100, Blue: -100
- ✓ Blue victory → Red: -100, Blue: +100
- ✓ Rewards sum to ~0 (zero-sum verified)

---

## What Changed

| Aspect | Old (Broken) | New (Fixed) |
|--------|-------------|-------------|
| **Rewards** | Always positive | Win/Loss (+100/-100) |
| **Win Condition** | Unclear | Explicit (state thresholds) |
| **Competition** | Cooperative | Adversarial |
| **Learning Signal** | Weak (±1.4 range) | Strong (±200 range) |
| **Zero-Sum** | ❌ Both positive | ✅ Sum = 0 |
| **Incentive** | "Do actions" | "Beat opponent" |

---

## Expected Training Results

### Generation 1-3:
- Win rate: ~50/50 (random exploration)
- Rewards: High variance (-100 to +100)
- Strategies: Random, exploratory

### Generation 4-7:
- Win rate: Fluctuating as agents adapt
- Rewards: Still high variance
- Strategies: Emerging patterns

### Generation 8-10:
- Win rate: Should stabilize around 50/50 (balanced)
- Rewards: Consistent high variance
- Strategies: Sophisticated adversarial behavior

**IF agents don't show improvement:**
- Check win rates (stuck at 100/0 = env too easy for one side)
- Check reward variance (low = weak signal)
- Check episode length (too short = no time to learn)

---

## Git Workflow

After training completes:

```bash
# Check status
git status

# Add new files
git add aceac_zerosum_environment.py
git add aceac_zerosum_training.py
git add ZERO_SUM_FIX_SUMMARY.md
git add models/zerosum/
git add logs/zerosum_training.json

# Commit
git commit -m "Implement true zero-sum adversarial reward system

- Replace broken cooperative rewards with proper win/loss structure
- Win: +100, Loss: -100, Draw: 0
- Clear adversarial dynamics with state-based win conditions
- Fresh agent training (not loading corrupted reward models)
- Validation confirms zero-sum property holds"

# Push
git push -u origin claude/fix-determinism-critical-01GmqCoxYLEnKRrUKrMMQAzX
```

---

## Troubleshooting

### "Red always wins alone"
- This is expected when Red acts without opposition
- Opponent is needed for realistic adversarial dynamics
- See: `aceac_zerosum_environment.py:374-422` for validation tests

### "Both agents get same reward"
- Check if using OLD environment (ACEACCoEvolutionEnv)
- Must use NEW environment (ZeroSumCyberEnv)

### "Rewards don't sum to zero"
- Small deviation (<5) is OK due to shaping bonus
- Large deviation (>10) means environment error

### "No learning happening"
- Check timesteps_per_generation (too low = insufficient learning)
- Check win conditions (too hard/easy = no gradient)
- Check entropy coefficient (too low = no exploration)

---

## Files Modified/Created

**New Files:**
- `aceac_zerosum_environment.py` - True zero-sum environment
- `aceac_zerosum_training.py` - Training script with proper rewards
- `ZERO_SUM_FIX_SUMMARY.md` - This document

**Existing Files (DO NOT USE for new training):**
- `aceac_coevolution_FIXED.py` - Still uses BROKEN rewards
- `aceac_dynamic_coevolution.py` - Still uses BROKEN rewards
- Pre-trained models in `models/` - Trained with BROKEN rewards

**Use ONLY:**
- `aceac_zerosum_training.py` for new training
- Models saved in `models/zerosum/` after training

---

## Success Criteria

After 10 generations, you should see:

1. **Win Rate Balance:** ~40-60% for each agent
2. **Reward Variance:** Std dev > 30
3. **Zero-Sum Property:** Mean(Red) + Mean(Blue) ≈ 0
4. **Strategic Diversity:** Different episodes have different outcomes
5. **Co-Evolution:** Win rates shift as agents adapt

**If all 5 criteria met:** System is working correctly ✅

**If criteria not met:** Reward structure still has issues ❌

---

Author: @sarowarzahan414
Date: 2025-11-19
Purpose: Fix fundamental adversarial training system
